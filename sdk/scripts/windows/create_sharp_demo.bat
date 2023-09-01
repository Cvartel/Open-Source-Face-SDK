@echo off

cd build
dotnet new sln -n csharp_face_demo
xcopy /isvy "..\samples\csharp\csharp_face_demo\" ".\csharp_face_demo"
xcopy /isvy "..\src\csharp_api" ".\csharp_face_demo\csharp_api"
dotnet sln csharp_face_demo.sln add csharp_face_demo/csharp_face_demo.csproj
cd csharp_face_demo
dotnet add package OpenCvSharp4
dotnet add package OpenCvSharp4.runtime.win
dotnet add package CommandLineParser
dotnet publish --configuration Release --output bin\publish /p:AllowUnsafeBlocks=true
cd ..
xcopy /isvy "csharp_face_demo\bin\publish" ".\make_install\bin\"

@echo Done!
