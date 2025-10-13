Install the `nbconvert[webpdf]` first.
```python
pip install "nbconvert[webpdf]"
```

Run the conversion
```python
jupyter nbconvert --to webpdf --allow-chromium-download filename.ipynb
```


