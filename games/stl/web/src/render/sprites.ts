// Frames are static PNGs under public/art/<character>/<pose>/0.png, already
// keyed, mirrored, and split. The Python sprite pipeline that prepared them is
// in git history at commit 938fb25 (src/arena/presentation/).

export type Character = "baku" | "hal" | "yakou";
export type Pose = "idle" | "dropping" | "seated" | "standing" | "win_screen";

const cache = new Map<string, HTMLImageElement>();
const failed = new Set<string>();

function key(character: Character, pose: Pose, index: number): string {
  return `${character}/${pose}/${index}`;
}

/**
 * A frame, or null while it loads or if the art is absent.
 *
 * A missing frame draws a labelled placeholder. Returning null rather than
 * throwing keeps a missing sprite from taking down the frame loop.
 */
export function frame(character: Character, pose: Pose, index: number): HTMLImageElement | null {
  const id = key(character, pose, index);
  if (failed.has(id)) return null;
  const cached = cache.get(id);
  if (cached) return cached.complete && cached.naturalWidth > 0 ? cached : null;

  const image = new Image();
  image.src = `/art/${id}.png`;
  image.addEventListener("error", () => failed.add(id));
  cache.set(id, image);
  return null;
}

/** Warm the frames a game actually uses, so the first paint is not empty. */
export function preload(): void {
  const players: Character[] = ["baku", "hal"];
  for (const character of players) {
    frame(character, "idle", 0);
    frame(character, "dropping", 0);
    frame(character, "seated", 0);
    frame(character, "win_screen", 0);
  }
  frame("yakou", "idle", 0);
  frame("yakou", "standing", 0);
}
