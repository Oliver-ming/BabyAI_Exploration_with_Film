#!/bin/zsh
set -e
cd "$(dirname "$0")"
# Close any stale open PowerPoint window holding an older cached copy of AIAA4220_Project3_Presentation.pptx
osascript -e 'tell application "Microsoft PowerPoint" to if it is running then quit saving no' >/dev/null 2>&1 || true
rm -f '~$AIAA4220_Project3_Presentation'*.pptx(N)
python3 scripts/make_case_figs.py > /dev/null
[[ -f assets/demo_attention_laws.gif ]] || python3 scripts/make_demo_gif.py
python3 scripts/make_plots.py > /dev/null
python3 build_ppt.py
scripts/export_pdf.sh AIAA4220_Project3_Presentation_Attention.pptx AIAA4220_Project3_Presentation_Attention.pdf
osascript -e 'tell application "Microsoft PowerPoint" to if it is running then quit saving no' >/dev/null 2>&1 || true
rm -f '~$AIAA4220_Project3_Presentation'*.pptx(N)
cp -f AIAA4220_Project3_Presentation_Attention.pdf AIAA4220_Project3_Presentation.pdf
rm -f renders/*.png(N)
pdftoppm -r 110 -png AIAA4220_Project3_Presentation_Attention.pdf renders/slide
ls renders
python3 scripts/make_script_md.py && python3 scripts/verify_deck.py
