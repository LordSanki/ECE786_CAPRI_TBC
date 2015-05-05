if [ -n $(uname -m | grep '64'| wc -l) != 0 ];
then
  dfile=$PWD/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
  if [ -e "$dfile" ];
  then
    echo "$dfile exists skipping download. Remove it to redownload"
  else
    wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
  fi
  if [ -d "cuda" ];
  then
    echo "direcoty $PWD/cuda exists skipping installation. Remove it to reinstall"
  else
    bash -lic "mkdir cuda; chmod u+x cudatoolkit_4.0.17_linux_64_ubuntu10.10.run; ./cudatoolkit_4.0.17_linux_64_ubuntu10.10.run -- -prefix=$PWD/cuda"
  fi
else
  dfile=$PWD/cudatoolkit_4.0.17_linux_32_ubuntu10.10.run
  if [ -e "$dfile" ];
  then
    echo "$dfile exists skipping download. Remove it to redownload"
  else
    wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_32_ubuntu10.10.run 
  fi
  if [ -d "cuda" ];
  then
    echo "direcoty $PWD/cuda exists skipping installation. Remove it to reinstall"
  else
    bash -lic "mkdir cuda; chmod u+x cudatoolkit_4.0.17_linux_32_ubuntu10.10.run; ./cudatoolkit_4.0.17_linux_32_ubuntu10.10.run -- -prefix=$PWD/cuda"
  fi
fi

sudo apt-get install build-essential
sudo apt-get install xutils-dev
sudo apt-get install gcc-4.4
sudo apt-get install g++-4.4
sudo apt-get install bison
sudo apt-get install flex
sudo apt-get install libz-dev

ln -sf /usr/bin/gcc-4.4 gcc
ln -sf /usr/bin/g++-4.4 g++

if [ -d "gpgpusim" ];
then
  bash -lic "cd gpgpusim; git pull;"
else
  git clone https://github.com/LordSanki/gpgpu-sim_distribution.git gpgpusim
  cp gpgpusim/setup_environment gpgpusim/setup_environment.back
fi

echo "export CUDA_INSTALL_PATH=$PWD/cuda" > gpgpusim/setup_environment
echo "export PATH=$PWD/:$PATH" >> gpgpusim/setup_environment
cat gpgpusim/setup_environment.back >> gpgpusim/setup_environment
bash -lic "cd gpgpusim; source setup_environment; make;"

echo "export PATH=$PWD:$PATH" > source_me
echo "export LD_LIBRARY_PATH=$PWD/gpgpusim/lib/"$(ls $PWD/gpgpusim/lib/)"/cuda-4000/release:$LD_LIBRARY_PATH" >>source_me
echo "export CUDA_INSTALL_PATH=$PWD/cuda" >> source_me
ln -sf $PWD/gpgpusim/version .
ln -sf $PWD/gpgpusim/configs/GTX480/config_fermi_islip.icnt .
ln -sf $PWD/gpgpusim/configs/GTX480/gpgpusim.config .
ln -sf $PWD/gpgpusim/configs/GTX480/gpuwattch_gtx480.xml .
echo "\n\n\nSetup Done. You can now compile and run code."
echo "Dont forget to source the source_me file.\n\n"

