if [$(uname -m | grep '64') != ""];
then
  if [ ! -a "cudatoolkit_4.0.17_linux_64_ubuntu10.10.run" ];
  then
    wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
    chmod u+x cudatoolkit_4.0.17_linux_64_ubuntu10.10.run
  fi
  if [ ! -d "cuda" ];
  then
    eval "mkdir cuda; ./cudatoolkit_4.0.17_linux_64_ubuntu10.10.run -- -prefix=$PWD/cuda"
  fi
else
  if [ ! -a "cudatoolkit_4.0.17_linux_32_ubuntu10.10.run" ];
  then
    wget http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_32_ubuntu10.10.run 
    chmod u+x cudatoolkit_4.0.17_linux_32_ubuntu10.10.run
  fi
  if [ ! -d "cuda" ];
  then
    eval "mkdir cuda; ./cudatoolkit_4.0.17_linux_32_ubuntu10.10.run -- -prefix=$PWD/cuda"
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

if [ ! -d "gpgpusim" ];
then
  git clone https://github.com/LordSanki/gpgpu-sim_distribution.git gpgpusim
  cp gpgpusim/setup_environment gpgpusim/setup_environment.back
else
  eval "cd gpgpusim; git pull;"
fi

echo "export CUDA_INSTALL_PATH=$PWD/../cuda" > gpgpusim/setup_environment
echo "export PATH=$PWD/../:$PATH" >> gpgpusim/setup_environment
cat gpgpusim/setup_environment.back >> gpgpusim/setup_environment
eval "cd gpgpusim; source setup_environment; make;"

echo "export PATH=$PWD:$PATH" > source_me
echo "export LD_LIBRARY_PATH=$PWD/gpgpusim/lib/"$(ls $PWD/gpgpusim/lib/)"/cuda-4000/release:$LD_LIBRARY_PATH" >>source_me
echo "export CUDA_INSTALL_PATH=$PWD/cuda" >> source_me
ln -sf $PWD/gpgpusim/version .
ln -sf $PWD/gpgpusim/configs/GTX480/config_fermi_islip.icnt .
ln -sf $PWD/gpgpusim/configs/GTX480/gpgpusim.config .
ln -sf $PWD/gpgpusim/configs/GTX480/gpuwattch_gtx480.xml .
echo "\n\n\nSetup Done. You can now compile and run code."
echo "Dont forget to source the source_me file.\n\n"

