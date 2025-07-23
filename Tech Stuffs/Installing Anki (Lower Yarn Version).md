Anki for arch linux require the older version of yarn of 1.x.x.

1. Downgrade the yarn version.
```bash
corepack prepare yarn@1.22.22 --activate
```
2. Install anki
```bash
yay -S anki
```
3. Switch back to the latest yarn version
```bash
corepack prepare yarn@stable --activate
```
