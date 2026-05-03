# Sample Documents

Place any PDF file here for testing. For the demo script, use:

```bash
# Option 1: Use your own PDF
cp /path/to/your/document.pdf samples/example.pdf
./scripts/demo.sh

# Option 2: Pass a PDF directly to the demo script
./scripts/demo.sh /path/to/your/document.pdf
```

For a quick test, download a small public-domain PDF:
```bash
curl -L -o samples/example.pdf "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
```
