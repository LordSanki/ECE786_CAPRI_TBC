sudo apt-get install build-essential
if [$(uname -m | grep '64') != ""];
then
  wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
  sudo chmod u+x cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
  echo "\n\n\nInstalling cuda. Press Enter when prompted for path\n\n"
  sudo ./cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
else
  wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_32_ubuntu10.10.run 
  sudo chmod u+x cudatoolkit_4.0.17_linux_32_ubuntu10.10.run
  echo "\n\n\nInstalling cuda. Press Enter when prompted for path\n\n"
  sudo ./cudatoolkit_4.0.17_linux_32_ubuntu10.10.run
fi

sudo apt-get install xutils-dev
sudo apt-get install gcc-4.4
sudo apt-get install g++-4.4
cd ../

git clone https://github.com/LordSanki/gpgpu-sim_distribution.git gpgpusim
cd gpgpusim
echo "export CUDA_INSTALL_PATH=/usr/local/cuda" >> setup_env
cat setup_environment >> setup_env
mv setup_env setup_environment
source setup_environment > /dev/null
make > /dev/null
cd ../

echo "export PATH=$PWD:$PATH" > source_me
echo "export LD_LIBRARY_PATH=$PWD/gpgpusim/lib/"$(ls $PWD/gpgpusim/lib/)"/cuda-4000/release:$LD_LIBRARY_PATH" >>source_me
echo "export CUDA_INSTALL_PATH=/usr/local/cuda/" >> source_me
ln -sf /usr/bin/gcc-4.4 gcc
ln -sf /usr/bin/g++-4.4 g++
ln -sf $PWD/../gpgpusim/version .
ln -sf $PWD/../gpgpusim/configs/GTX480/* .
echo "\n\n\nSetup Done. You can now compile and run code."
echo "Dont forget to source the source_me file.\n\n"
