# Sheet Data Image/PDF-to-CSV Converter

## Dependencies
```
pip install pdf2image
pip install opencv-python
pip install pytesseract
```
This tool also require Pillow library, this could install through conda:
```
conda install -c conda-forge pillow
```
## Usage
```
python spreadsheet_parser.py [input-file] [output-file] [language]
```
`input-file`: Input file, must be pdf file or image file  
`output-file`: Output path for csv output  
`language`: content language (must be pre-install tessdata model for that language)


Example:
```
python spreadsheet_parser.py examples/test.png output.csv
```
