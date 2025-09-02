
# nvim-ros2 Plugin  

https://github.com/ErickKramer/nvim-ros2
The **nvim-ros2** plugin integrates ROS2 tools into Neovim. It provides Telescope pickers for quickly exploring **nodes, topics, services, actions, and interfaces** inside a ROS2 workspace.  

## Features
- Telescope integration for browsing ROS2 entities.  
- Treesitter and Plenary support.  
- Lazy-loaded for performance (`VeryLazy`).  

## Setup
Plugin configuration (already in your `lazy.nvim` setup):

```lua
{
  "ErickKramer/nvim-ros2",
  event = "VeryLazy",
  dependencies = {
    "nvim-lua/plenary.nvim",
    "nvim-telescope/telescope.nvim",
    "nvim-treesitter/nvim-treesitter",
  },
  opts = { autocmds = true, telescope = true, treesitter = true },
  config = function(_, opts)
    require("nvim-ros2").setup(opts)
    pcall(function()
      require("telescope").load_extension("ros2")
    end)
  end,
  keys = {
    { "<leader>rn", "<cmd>Telescope ros2 nodes<CR>", desc = "ROS2 nodes" },
    { "<leader>rt", "<cmd>Telescope ros2 topics<CR>", desc = "ROS2 topics" },
    { "<leader>rs", "<cmd>Telescope ros2 services<CR>", desc = "ROS2 services" },
    { "<leader>ra", "<cmd>Telescope ros2 actions<CR>", desc = "ROS2 actions" },
    { "<leader>ri", "<cmd>Telescope ros2 interfaces<CR>", desc = "ROS2 interfaces" },
  },
}
```

## Hotkeys

| Keybinding        | Command                          | Description       |
|-------------------|----------------------------------|-------------------|
| `<leader>rn`      | `:Telescope ros2 nodes`          | List ROS2 nodes   |
| `<leader>rt`      | `:Telescope ros2 topics`         | List ROS2 topics  |
| `<leader>rs`      | `:Telescope ros2 services`       | List ROS2 services|
| `<leader>ra`      | `:Telescope ros2 actions`        | List ROS2 actions |
| `<leader>ri`      | `:Telescope ros2 interfaces`     | List ROS2 interfaces |

## Checking Dependencies

To ensure that **Telescope**, **nvim-treesitter**, and **Plenary** are enabled:

1. **Telescope**  
   Run:  
   ```vim
   :Telescope
   ```  
   If the command is not found, Telescope is not enabled.  

2. **nvim-treesitter**  
   Run:  
   ```vim
   :TSInstallInfo
   ```  
   If the command is not recognized, Treesitter is not enabled.  

3. **Plenary**  
   Run inside Neovim’s Lua prompt (`:lua`):  
   ```lua
   =require("plenary")
   ```  
   If it throws `module 'plenary' not found`, Plenary is not enabled.  

## Enabling via Lazy Extras

If any of these plugins are missing, you can enable them via [LazyVim extras](https://www.lazyvim.org/extras/):

```lua
{ import = "lazyvim.plugins.extras.ui.telescope" },
{ import = "lazyvim.plugins.extras.lang.treesitter" },
{ import = "lazyvim.plugins.extras.util.plenary" },
```

Add these lines to your `lazy.lua` plugin spec.  

## Usage
Open Neovim inside your ROS2 project workspace and use the hotkeys above to browse the corresponding ROS2 resources.  

