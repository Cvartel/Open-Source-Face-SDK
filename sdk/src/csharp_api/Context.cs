using System;
using System.Collections.Generic;
using CSharpApi;
using System.Text;
using System.Runtime.InteropServices;
using System.Collections;
using System.Reflection.Metadata.Ecma335;

namespace CSharpApi
{
    public class Context : ComplexObject , IEnumerable, IEnumerator
    {
        private bool _weak;
        private object? data;
        private ulong n;

        object IEnumerator.Current => GetByIndex((int)n-1);

        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void* TDVContext_create(ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void TDVContext_destroy(void* ctx, ref void* eh);
        
        unsafe public Context(void* the_impl = null) : base(the_impl)
        {
            _weak = false;
            data = null;
            if (the_impl == null)
            {   
                void* exception = ErrorMethods.MakeException();
                _impl = TDVContext_create(ref exception);
                ErrorMethods.CheckException(exception);
            }
        }
        
        unsafe ~Context()
        {
            if(!_weak)
            {
                void* exception = ErrorMethods.MakeException();
                TDVContext_destroy(_impl, ref exception);

                ErrorMethods.CheckException(exception); 
            }
        }
        
        public void Invoke(object ctx)
        {
            if (ctx is string strValue)
            {
                StringParser(strValue);
            }
            else if (ctx is long longValue)
            {
                LongParser(longValue);  
            }
            else if (ctx is int intValue)
            {
                long longval = (long)intValue;
                LongParser(longval);
            }
            else if (ctx is double doubleValue)
            {
                DoubleParser(doubleValue);
            }
            else if (ctx is float floatValue)
            {
                double doubleval = (double)floatValue;
                DoubleParser(doubleval);
            }
            else if (ctx is bool boolValue)
            {
                BoolParser(boolValue);
            }
            else if (ctx is byte[] byteArrayValue)
            {
                ByteParser(byteArrayValue);
            }
            else if (ctx is Dictionary<object,object> dict)
            {
                DictionaryParser(dict);
            }
            else if (ctx is List<object> listValue)
            {
                ListParser(listValue);
            }
        }
        public Context this[int index]
        {
            get 
            {return GetByIndex(index);}
        }
        public Context this[string key]
        {
            get{return GetByKey(key);}
            set{value = GetOrInsertByKey(key);}
        }
        public Context GetItem(int key)
        {
            return GetByIndex(key);
        }
        public bool MoveNext()
        {
            if (n - 1 < GetLength() - 1)
            {
                n++;
                return true;
            }
            else
            return false;
        }
        public void Reset()
        {
            Iter();
        }
        public Context GetItem(string key)
        {
            return GetByKey(key);
        }

        public ulong Length()
        {
            return GetLength();
        }

        public void Iter()
        {
            this.n = 0;
        }

        unsafe public Context Next()
        {
            if (this.n >= GetLength())
            {
                throw new IndexOutOfRangeException();
            }
            Context result = GetByIndex((int)n);
            n += 1;
            return result;
        }

        public void SetWeak(bool weak)
        {
            _weak = weak;
        }
        
        public void Push_Back(Context data)
        {
            PushBack(data);
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void TDVContext_putStr(void* ctx, string str, ref void* eh);
        
        unsafe public void SetStr(string value)
        {
            void* exception = ErrorMethods.MakeException();

            TDVContext_putStr(_impl, value, ref exception);

            ErrorMethods.CheckException(exception);
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern string TDVContext_getStr(void* ctx, char* buff, ref void* eh);
  
        unsafe public string GetStr()
        {
            void* exception = ErrorMethods.MakeException();
            char* buff = null;
            string value = TDVContext_getStr(_impl, buff, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void TDVContext_putDouble(void* ctx, double val, ref void* eh);
  
        unsafe public void SetDouble(double value)
        {
            void* exception = ErrorMethods.MakeException();

            TDVContext_putDouble(_impl, (double)value, ref exception);

            ErrorMethods.CheckException(exception);
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern double TDVContext_getDouble(void* ctx, ref void* eh);
  
        unsafe public double GetDouble()
        {
            void* exception = ErrorMethods.MakeException();
            double value = TDVContext_getDouble(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void TDVContext_putBool(void* ctx, bool val, ref void* eh);
  
        unsafe public void SetBool(bool value)
        {
            void* exception = ErrorMethods.MakeException();

            TDVContext_putBool(_impl, value, ref exception);

            ErrorMethods.CheckException(exception);
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern char* TDVContext_putDataPtr(void* ctx, byte* str, ulong copy_sz ,ref void* eh);
  
        unsafe public void SetDataPtr(byte[] value)
        {
            void* exception = ErrorMethods.MakeException();
            fixed (byte* ch = value)
            {
                ulong length = (ulong)value.Length;
                TDVContext_putDataPtr(_impl, ch, length, ref exception);
                ErrorMethods.CheckException(exception); 
            }
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern byte* TDVContext_getDataPtr(void* ctx, ref void* eh);
  
        unsafe public byte* GetDataPtr()
        {
            void* exception = ErrorMethods.MakeException();

            byte* result = TDVContext_getDataPtr(_impl,ref exception);
            ErrorMethods.CheckException(exception);

            return result;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_getBool(void* ctx ,ref void* eh);
  
        unsafe public bool GetBool()
        {
            void* exception = ErrorMethods.MakeException();

            bool value = TDVContext_getBool(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern char* TDVContext_putLong(void* ctx, long val, ref void* eh);
  
        unsafe public void SetLong(long value)
        {
            void* exception = ErrorMethods.MakeException();

            TDVContext_putLong(_impl, value, ref exception);

            ErrorMethods.CheckException(exception);
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern long TDVContext_getLong(void* ctx, ref void* eh);
        unsafe public long GetLong()
        {
            void* exception = ErrorMethods.MakeException();

            long value = TDVContext_getLong(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern ulong TDVContext_getUnsignedLong(void* ctx, ref void* eh);
        unsafe public ulong GetUnsignedLong()
        {
            void* exception = ErrorMethods.MakeException();

            ulong value = TDVContext_getUnsignedLong(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern ulong TDVContext_getLength(void* ctx, ref void* eh);
        unsafe public ulong GetLength()
        {
            void* exception = ErrorMethods.MakeException();

            ulong value = TDVContext_getLength(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern ulong TDVContext_getStrSize(void* ctx, ref void* eh);
  
        unsafe public ulong GetStrSize()
        {
            void* exception = ErrorMethods.MakeException();

            ulong value = TDVContext_getStrSize(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        [DllImport("open_source_sdk.dll", CharSet = CharSet.Ansi)]
        unsafe private static extern void* TDVContext_getOrInsertByKey(void* ctx, string key ,ref void* eh);
        unsafe public Context GetOrInsertByKey(string key)
        {
            void* exception = ErrorMethods.MakeException();

            void* new_impl = TDVContext_getOrInsertByKey(_impl, key, ref exception);

            ErrorMethods.CheckException(exception);
            Context ctx = new Context(new_impl);
            ctx.SetWeak(true);

            return ctx;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void* TDVContext_getByKey(void* ctx, string key ,ref void* eh);
        unsafe public Context GetByKey(string key)
        {
            void* exception = ErrorMethods.MakeException();
            void* new_impl = TDVContext_getByKey(_impl, key, ref exception);

            ErrorMethods.CheckException(exception);
            Context ctx = new Context(new_impl);
            ctx.SetWeak(true);
            return ctx;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void* TDVContext_getByIndex(void* ctx, int index ,ref void* eh);
        unsafe public Context GetByIndex(int index)
        {
            void* exception = ErrorMethods.MakeException();
            void* new_impl = TDVContext_getByIndex(_impl, index, ref exception);

            ErrorMethods.CheckException(exception);
            Context ctx = new Context(new_impl);
            ctx.SetWeak(true);
            return ctx;
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void* TDVContext_clone(void* ctx, ref void* eh);
  
        unsafe public Context Clone()
        {
            void* exception = ErrorMethods.MakeException();

            void* impl =  TDVContext_clone(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return new Context(impl);
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void TDVContext_pushBack(void* handle, void* data , bool copy, ref void* eh);
        unsafe public void PushBack(Context data)
        {
            void* exception = ErrorMethods.MakeException();

            TDVContext_pushBack(_impl, data._impl, true, ref exception);
            
            ErrorMethods.CheckException(exception);
        }
        unsafe public void DictionaryParser(Dictionary<object,object> ctx)
        {
            foreach(var key in ctx.Keys)
            {
                GetOrInsertByKey((string)key).Invoke(ctx[key]);
            }
        }
        unsafe public void ListParser(List<object> ctx)
        {
            foreach (var value in ctx)
            {
                Context ctt = new Context();
                if (value is string strValue)
                {
                    ctt.StringParser(strValue);
                }
                else if (value is long longValue)
                {
                    ctt.LongParser(longValue);
                }
                else if (value is int intValue)
                {
                    ctt.LongParser((long)intValue);
                }
                else if (value is double doubleValue)
                {
                    ctt.DoubleParser(doubleValue);
                }
                else if (value is float floatValue)
                {
                    ctt.DoubleParser((double)floatValue);
                }
                else if (value is bool boolValue)
                {
                    ctt.BoolParser(boolValue);
                }
                else if (value is byte[] byteArrayValue)
                {
                    ctt.ByteParser(byteArrayValue);
                }
                PushBack(ctt);
            }
        }
        unsafe public void StringParser(string ctx)
        {
            SetStr(ctx);
        }
        unsafe public void LongParser(long ctx) 
        {
            SetLong(ctx);
        }
        unsafe public void DoubleParser(double ctx)
        {
            SetDouble(ctx);
        }
        unsafe public void BoolParser(bool ctx)
        {
            SetBool(ctx);
        }
        unsafe public void ByteParser(byte[] ctx)
        {
            SetDataPtr(ctx);
        }
        
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern IntPtr TDVContext_getKeys(void* handle, ulong length, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern void TDVContext_freePtr(IntPtr ptr);
   
        unsafe public List<string> GetKeys()
        {
            void* exception = ErrorMethods.MakeException();
            ulong cout_keys = GetLength();
            IntPtr p_value_array = TDVContext_getKeys(_impl, cout_keys, ref exception);
            string[] strings = new string[cout_keys];
            for (ulong i = 0; i < cout_keys; i++)
            {
                IntPtr charPtr = Marshal.ReadIntPtr(p_value_array, (int)i * IntPtr.Size);
                strings[i] = Marshal.PtrToStringAnsi(charPtr);
            }
            ErrorMethods.CheckException(exception);
            List<string> result = new List<string>();
            for (int i = 0; i < strings.Length; i++)
            {
                result.Add(strings[i]);
            }
            for (ulong i = 0; i < cout_keys; i++)
            {
                IntPtr charPtr = Marshal.ReadIntPtr(p_value_array, (int)i * IntPtr.Size);
                TDVContext_freePtr(charPtr);
            }
            TDVContext_freePtr(p_value_array);

            return result;
        }
        public List<string> Keys()
        {
            return GetKeys();
        }
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isNone(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isArray(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isObject(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isBool(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isLong(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isUnsignedLong(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isDouble(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isString(void* ctx, ref void* eh);
        [DllImport("open_source_sdk.dll")]
        unsafe private static extern bool TDVContext_isDataPtr(void* ctx, ref void* eh);
        unsafe public bool IsNone()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isNone(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsArray()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isArray(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsObject()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isObject(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsBool()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isBool(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsLong()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isLong(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsUnsignedLong()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isUnsignedLong(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsDouble()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isDouble(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsString()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isString(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public bool IsDataPtr()
        {
            void* exception = ErrorMethods.MakeException();
            bool value = TDVContext_isDataPtr(_impl, ref exception);

            ErrorMethods.CheckException(exception);
            return value;
        }
        unsafe public object GetValue()
        {
            if (IsNone())
                return null;
            if (IsBool())
                return GetBool();
            if (IsString())
                return GetStr();
            if (IsLong())
                return GetLong();
            if (IsUnsignedLong())
                return GetUnsignedLong();
            if (IsDouble())
                return GetDouble();
            if (IsDataPtr())
                return (IntPtr)GetDataPtr();
            return null;
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return (IEnumerator)this;
        }
    }
}