// The ONE place where camera coordinates become screen coordinates.
// Backend coordinates are normalised [0,1] in the UN-mirrored camera frame.
// The <video> is displayed with object-fit: cover and (optionally) mirrored with CSS.

export function coverTransform(videoW, videoH, boxW, boxH, mirrored) {
  if (!videoW || !videoH || !boxW || !boxH) return null;
  const scale = Math.max(boxW / videoW, boxH / videoH);   // object-fit: cover
  const dispW = videoW * scale, dispH = videoH * scale;
  const offX = (boxW - dispW) / 2, offY = (boxH - dispH) / 2;
  return {
    scale, offX, offY, dispW, dispH, mirrored,
    // normalised camera -> CSS px inside the box
    toScreen(xn, yn) {
      const x = mirrored ? 1 - xn : xn;
      return [offX + x * dispW, offY + yn * dispH];
    },
    // CSS px inside the box -> normalised camera (inverse, used for the framing guide)
    toCamera(px, py) {
      let xn = (px - offX) / dispW;
      if (mirrored) xn = 1 - xn;
      return [xn, (py - offY) / dispH];
    },
    // visible part of the camera image in normalised camera coords
    visibleCameraRect() {
      const a = this.toCamera(0, 0), b = this.toCamera(boxW, boxH);
      return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])];
    },
  };
}
