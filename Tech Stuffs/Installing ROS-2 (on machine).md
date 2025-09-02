# Installing ROS-2

## Ensure dependencies are installed
```bash
sudo pacman -S --needed base-devel git cmake
```

## Install RosDep
```bash
yay -S python-rosdep
```

## Init Rosdep
```bash
sudo rosdep init
```

```bash
rosdep update
```

## Install Ros-Humble
```bash
yay -S ros2-humble
```
Note that latest version are not available on AUR