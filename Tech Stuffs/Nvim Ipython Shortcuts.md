## Quick-start flow 

| Mode | Keys                        | Action                                                                                                    |
| ---- | --------------------------- | --------------------------------------------------------------------------------------------------------- |
| N    | `<leader>np`                | Convert between `.ipynb` ↔ script; opens in `jupyter-notebook` after converting; preserves saved outputs. |
| N    | `<leader>os`                | Open output split and start shell (default `ipython3`).                                                   |
| N    | `<CR>`                      | Send current **line** to the output split.                                                                |
| V    | `<CR>`                      | Send **visual selection** to the output split.                                                            |
| N    | `<leader><space>`           | Send **current cell** to the output split (cell executions are what get saved).                           |
| N    | `<leader>co` / `<leader>cO` | Create **code cell** **below / above**.                                                                   |
| N    | `<leader>ct` / `<leader>cT` | Create **text/markdown cell** **below / above**.                                                          |
| N    | `<leader>ck` / `<leader>cj` | **Move** current cell **up / down**.                                                                      |
| N    | `<leader>cs`                | **Split** current cell.                                                                                   |
| N    | `<leader>cm` / `<leader>cM` | **Merge** current cell with **below / above**.                                                            |
| N    | `<leader>hs`                | Open **output-history** split.                                                                            |
| N    | `<leader>so`                | Show **saved output** of the **current cell** in the history split.                                       |
| N    | `<leader>j` / `<leader>k`   | Scroll **down / up** in the output-history split.                                                         |
| N    | `<leader>hd`                | Close the output-history split.                                                                           |
| N    | `<leader>np`                | Convert back to `.ipynb` (same key); opens in `jupyter-notebook`.                                         |

## Other useful / less-used keybinds

| Mode | Keys | Action |
|---|---|---|
| N | `<leader>ohs` | Open **output** and **history** splits together. |
| N | `<leader>od` | Close output split. |
| N | `<leader>ohd` | Close **both** output + history splits (asks for confirm by default). |
| N | `<leader>ts` | Open output split **without** starting a command (bare terminal). |
| N | `<leader>ah` | Toggle auto-show of saved outputs on `CursorHold`. |
| N | `<leader>sl` | Apply window **layout** from `g:jukit_layout`. |
| N | `<leader>cc` | Execute **all cells up to** current cell. |
| N | `<leader>all` | Execute **all cells**. |
| N | `<leader>cd` | **Delete** current cell. |
| N | `<leader>J` / `<leader>K` | **Jump** to **next / previous** cell. |
| N | `<leader>ddo` / `<leader>dda` | Delete saved outputs: **current cell / all cells**. |
| N | `<leader>ht` / `<leader>rht` | Save to **HTML** (open) / **rerun all then HTML** (open). |
| N | `<leader>pd` / `<leader>rpd` | Save to **PDF** (open) / **rerun all then PDF** (open). |
| N | `<leader>pos` | Set **überzug** preview window position/size (if using überzug). |
| Cmd | `:JukitOut {cmd}` | Run `{cmd}` (e.g., activate venv) then open output split and start shell. |
| Cmd | `:JukitOutHist {cmd}` | Same as above **and** open output-history split. |

## Checking if JSON is okie
```bash
python -m json.tool numpy_basics.ipynb >/dev/null && echo "JSON OK"
```

## Rewriting into correct notebook
```bash
jupyter nbconvert --to notebook --nbformat 4 \
  --output numpy_basics_fixed.ipynb \
  numpy_basics.ipynb
```
