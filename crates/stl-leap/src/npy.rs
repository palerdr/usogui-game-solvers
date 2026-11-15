//! Version-1 `.npy` files: memory-mapped little-endian arrays in C order.
//!
//! The leap artifact stores every table in this format, so the Python reference can read what
//! this crate writes and the reverse. The header writer matches `numpy.lib.format` byte for byte.

use crate::{Error, io};
use memmap2::{Mmap, MmapMut};
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

const MAGIC: &[u8] = b"\x93NUMPY\x01\x00";

const _: () = assert!(
    cfg!(target_endian = "little"),
    "the tables are little-endian"
);

/// An element type of a table: `f64` (`<f8`) or `u8` (`|u1`).
pub(crate) trait Element: Copy {
    const DESCR: &'static str;
}

impl Element for f64 {
    const DESCR: &'static str = "<f8";
}

impl Element for u8 {
    const DESCR: &'static str = "|u1";
}

/// A read-only memory map of one `.npy` array.
pub(crate) struct Npy {
    map: Mmap,
    offset: usize,
    pub(crate) shape: Vec<usize>,
}

/// A writable memory map of one `.npy` array.
pub(crate) struct NpyMut {
    map: MmapMut,
    offset: usize,
    path: PathBuf,
}

fn format(path: &Path, reason: impl Into<String>) -> Error {
    Error::Format {
        path: path.to_path_buf(),
        reason: reason.into(),
    }
}

/// Return the header that numpy writes for a C-order array of this type and shape.
fn header<T: Element>(shape: &[usize]) -> Vec<u8> {
    let dims = match shape {
        [n] => format!("({n},)"),
        _ => format!(
            "({})",
            shape
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut text = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {dims}, }}",
        T::DESCR
    );
    // numpy leaves room for the first axis to grow to 21 digits.
    text.push_str(&" ".repeat(21 - shape[0].to_string().len()));
    let padding = 64 - (MAGIC.len() + 2 + text.len() + 1) % 64;
    text.push_str(&" ".repeat(padding));
    text.push('\n');
    let mut bytes = MAGIC.to_vec();
    bytes.extend_from_slice(
        &u16::try_from(text.len())
            .expect("a short header")
            .to_le_bytes(),
    );
    bytes.extend_from_slice(text.as_bytes());
    bytes
}

/// Parse a version-1 header and return the body offset and shape of a `T` array.
fn parse<T: Element>(path: &Path, bytes: &[u8]) -> Result<(usize, Vec<usize>), Error> {
    if bytes.len() < 10 || &bytes[..8] != MAGIC {
        return Err(format(path, "not a version-1 .npy file"));
    }
    let offset = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    let text = std::str::from_utf8(
        bytes
            .get(10..offset)
            .ok_or_else(|| format(path, "short header"))?,
    )
    .map_err(|_| format(path, "header is not text"))?;
    let field = |name: &str, end: char| {
        let start = text.find(name)? + name.len();
        Some(&text[start..start + text[start..].find(end)?])
    };
    let descr = field("'descr': '", '\'');
    if descr != Some(T::DESCR) || field("'fortran_order': ", ',') != Some("False") {
        return Err(format(
            path,
            format!("{descr:?} is not a C-order {} array", T::DESCR),
        ));
    }
    let shape = field("'shape': (", ')')
        .ok_or_else(|| format(path, "header has no shape"))?
        .split(',')
        .map(str::trim)
        .filter(|x| !x.is_empty())
        .map(str::parse)
        .collect::<Result<Vec<usize>, _>>()
        .map_err(|_| format(path, "bad shape"))?;
    let body = shape.iter().product::<usize>() * size_of::<T>();
    if offset % 8 != 0 || bytes.len() != offset + body {
        return Err(format(path, "file length disagrees with the header"));
    }
    Ok((offset, shape))
}

/// View aligned little-endian bytes as elements.
#[allow(unsafe_code)]
fn cast<T: Element>(bytes: &[u8]) -> &[T] {
    assert!(bytes.as_ptr().cast::<T>().is_aligned() && bytes.len().is_multiple_of(size_of::<T>()));
    // SAFETY: the body of a memory map starts at a page boundary plus a multiple of eight, the
    // length is a whole number of elements, and every bit pattern is a valid f64 or u8.
    unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast(), bytes.len() / size_of::<T>()) }
}

#[allow(unsafe_code)]
fn cast_mut<T: Element>(bytes: &mut [u8]) -> &mut [T] {
    assert!(bytes.as_ptr().cast::<T>().is_aligned() && bytes.len().is_multiple_of(size_of::<T>()));
    // SAFETY: as in `cast`; the exclusive borrow of the map makes the view unique.
    unsafe {
        std::slice::from_raw_parts_mut(bytes.as_mut_ptr().cast(), bytes.len() / size_of::<T>())
    }
}

impl Npy {
    /// Map an existing array of element type `T`.
    #[allow(unsafe_code)]
    pub(crate) fn open<T: Element>(path: &Path) -> Result<Self, Error> {
        let file = File::open(path).map_err(io(path))?;
        // SAFETY: the artifact files are written once and never modified while mapped.
        let map = unsafe { Mmap::map(&file) }.map_err(io(path))?;
        let (offset, shape) = parse::<T>(path, &map)?;
        Ok(Self { map, offset, shape })
    }

    /// Return the array body in C order.
    pub(crate) fn data<T: Element>(&self) -> &[T] {
        cast(&self.map[self.offset..])
    }
}

impl NpyMut {
    /// Create a zero-filled array file, replacing any file at `path`.
    #[allow(unsafe_code)]
    pub(crate) fn create<T: Element>(path: &Path, shape: &[usize]) -> Result<Self, Error> {
        let head = header::<T>(shape);
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(io(path))?;
        let length = head.len() + shape.iter().product::<usize>() * size_of::<T>();
        file.set_len(length as u64).map_err(io(path))?;
        // SAFETY: this process created the file and is its only writer.
        let mut map = unsafe { MmapMut::map_mut(&file) }.map_err(io(path))?;
        map[..head.len()].copy_from_slice(&head);
        Ok(Self {
            map,
            offset: head.len(),
            path: path.to_path_buf(),
        })
    }

    /// Return the writable array body in C order.
    pub(crate) fn data<T: Element>(&mut self) -> &mut [T] {
        cast_mut(&mut self.map[self.offset..])
    }

    /// Write the array to disk.
    pub(crate) fn flush(&self) -> Result<(), Error> {
        self.map.flush().map_err(io(&self.path))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_match_numpy() {
        // numpy 2.5 writes these two headers for the shapes of an H table and a packed bitmap.
        let dense = header::<f64>(&[17011, 17012]);
        assert_eq!(dense.len(), 128);
        assert!(dense.starts_with(b"\x93NUMPY\x01\x00\x76\x00{'descr': '<f8', 'fortran_order': False, 'shape': (17011, 17012), }"));
        let bits = header::<u8>(&[36_171_766]);
        assert_eq!(bits.len(), 128);
        assert!(bits.ends_with(b" \n"));
        assert!(String::from_utf8_lossy(&bits).contains("'shape': (36171766,), }"));
    }
}
