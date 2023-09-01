using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
namespace CSharpApi
{
    public class Service
    {
        protected string _pathToDir;
        public Service(string pathToDir)
        {
            _pathToDir = pathToDir;
        }

        public static Service CreateService(string pathToDir)
        {
            string _libPath = "";
            if(RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            {
                _libPath = string.Format("\\bin\\open_source_sdk.dll");
            }
            else
            {
                _libPath = string.Format("/lib/libopen_source_sdk.so");
            }
            string _dllPath = pathToDir + _libPath; //по сути ничего не делает
            
            return new Service(pathToDir);
        }
        unsafe public ProcessingBlock CreateProcessingBlock(Dictionary<object, object> ctx)
        {
            ctx["@sdk_path"] = _pathToDir;
            if(!ctx.ContainsKey("ONNXRuntime"))
            {
                ctx["ONNXRuntime"] = new Dictionary<object, object>
                {
                   { "library_path", _pathToDir + (Environment.OSVersion.Platform == PlatformID.Win32NT ? "\\bin" : "/lib") }
                };
            }
            return new ProcessingBlock(ctx);
        }
        unsafe public Context CreateContext(object ctx)
        {
            Context ctr = new Context();
            ctr.Invoke(ctx);
            return ctr;
        }
    }
}
