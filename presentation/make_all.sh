#!/bin/zsh
set -e
cd "$(dirname "$0")"
python3 scripts/make_plots.py > /dev/null
python3 scripts/make_case_figs.py > /dev/null
python3 scripts/make_demo_gif.py
python3 build_ppt.py
scripts/export_pdf.sh AIAA4220_Project3_Presentation.pptx AIAA4220_Project3_Presentation.pdf
rm -f renders/*.png(N)
pdftoppm -r 110 -png AIAA4220_Project3_Presentation.pdf renders/slide
ls renders
python3 scripts/make_script_md.py && python3 scripts/verify_deck.py
