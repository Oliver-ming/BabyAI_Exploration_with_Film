#!/bin/zsh
set -e
cd "$(dirname "$0")"
python3 scripts/make_plots.py > /dev/null
python3 build_ppt.py
scripts/export_pdf.sh AIAA4220_Project3_Presentation_Attention.pptx AIAA4220_Project3_Presentation_Attention.pdf
cp -f AIAA4220_Project3_Presentation_Attention.pdf AIAA4220_Project3_Presentation.pdf
rm -f renders/*.png(N)
pdftoppm -r 110 -png AIAA4220_Project3_Presentation_Attention.pdf renders/slide
ls renders
python3 scripts/make_script_md.py && python3 scripts/verify_deck.py
