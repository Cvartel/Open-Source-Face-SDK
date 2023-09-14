#!/bin/bash

mkdir -p build
cd build
dotnet new sln -n csharp_face_demo

cp -r ../samples/csharp/csharp_face_demo ./csharp_face_demo

cp ../src/csharp_api/* ./csharp_face_demo/

cd ./csharp_face_demo/
find . -type f -exec sed -i "s@open_source_sdk.dll@libopen_source_sdk.so@g" {} +
cd ../

dotnet sln csharp_face_demo.sln add csharp_face_demo/csharp_face_demo.csproj

cd csharp_face_demo

dotnet add package OpenCvSharp4
dotnet add package OpenCvSharp4_.runtime.ubuntu.20.04-x64 #Replace for correct Ubuntu version and CPU architecture. Find at https://www.nuget.org/packages
dotnet add package CommandLineParser

dotnet publish --configuration Release --output bin/publish /p:AllowUnsafeBlocks=true

cd ..
cp -r csharp_face_demo/bin/publish/* ./make-install/bin
cp ./make-install/lib/libopen_source_sdk.so ./make-install/bin/

echo "Done!"

