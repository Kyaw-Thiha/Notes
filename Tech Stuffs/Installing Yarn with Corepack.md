Installing yarn through corepack will yield package dependency errors from yay/pacman.

To fix this, we must 'fake' a yarn installation.

1. Remove existing installations with 
   ```bash
	sudo rm /usr/bin/yarn /usr/bin/yarnpkg
	```
2. Uninstall yarn from yay/pacman (in case you have it) with 
   ```bash
	sudo pacman -R yarn
	```
4. To prevent yay/pacman from trying to install yarn, create a fake package.
	```bash
	mkdir -p ~/builds/dummy-yarn && cd ~/builds/dummy-yarn
    ```
	```bash
	nano PKGBUILD
    ```
    Then, paste the following inside PKGBUILD -
    ```
    pkgname=dummy-yarn
	pkgver=1.0
	pkgrel=1
	pkgdesc="Dummy package to satisfy yarn dependency (Corepack is used instead)"
	arch=('any')
	provides=('yarn')
	conflicts=('yarn')
	license=('MIT')
	package() { 
	  mkdir -p "$pkgdir"
	}
	```
5. Exit nano, and go back to the terminal.
   Ensure you are in `~/builds/dummy-yarn`.
   Then, run the following -
   ```bash
	makepkg -si
	```
6. Ensure npm is already installed on your machine. 
   You can check for it with 
   ```bash
	node --version
	```
7. Enable the corepack yarn with 
   ```bash
	sudo corepack enable
	corepack prepare yarn@stable --activate  
     ```
8. To ensure yarn is installed corrected, check its version with `yarn --version`. 
   It should display something greater than 1 like 3.x.x
