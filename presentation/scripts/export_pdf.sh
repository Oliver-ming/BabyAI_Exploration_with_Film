#!/bin/zsh
# usage: export_pdf.sh in.pptx out.pdf   (uses Microsoft PowerPoint via AppleScript)
IN="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"
OUT="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
rm -f "$OUT"
osascript <<OSA
tell application "Microsoft PowerPoint"
  set pres to open POSIX file "$IN"
  delay 2
  save active presentation in POSIX file "$OUT" as save as PDF
  close active presentation saving no
end tell
OSA
ls -la "$OUT"
