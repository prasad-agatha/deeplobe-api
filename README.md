# deeplobe-api

# before docker build

- install nvidia-docker2

  `sudo apt-get install -y nvidia-docker2`

- Update cuda to 11.3

  - Clear nvidia and cuda

    ```sudo apt-get purge nvidia*
    sudo apt-get autoremove
    sudo apt-get autoclean
    sudo rm -rf /usr/local/cuda*
    ```

  - Install nvidia-driver

    ```sudo apt update
    sudo apt upgrade
    sudo apt install nvidia-driver-495
    sudo reboot
    ```

  - Install cuda

    ```wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    sudo apt-get update
    sudo apt-get -y install cuda
    ```
