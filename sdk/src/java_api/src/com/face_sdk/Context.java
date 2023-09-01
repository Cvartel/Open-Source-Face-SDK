package com.face_sdk;


public class Context {
	public long context_ptr;
	public boolean weak;
	static {
		System.loadLibrary("open_source_sdk_jni");
	}

	Context(long context_ptr, boolean weak){
		this.context_ptr = context_ptr;
		this.weak = weak;
	}

	public void dispose() {
		if (!weak)
			destroyContext_jni();
	}

	public void finalize() {
		dispose();
	}

	public Context getByIndex(int key){
		long ptr = getByIndex_jni(key);
		return new Context(ptr, true);
	}

	public Context getByKey(String key){
		long ptr = getByKey_jni(key);
		return new Context(ptr, true);
	}

	public Context getOrInsertByKey(String key){
		long ptr = getOrInsertByKey_jni(key);
		return new Context(ptr, true);
	}

	public void pushBack(Context data){
		pushBack_jni(data, true);
	}

	public native void destroyContext_jni();

	public native long getByIndex_jni(int key);
	public native long getByKey_jni(String key);
	public native long getOrInsertByKey_jni(String key);

	public native void copy_jni(Context dst);
	public native long clone_jni();
	public native void clear();                  //jni

	public native void setLong(long value);      //jni
	public native void setDouble(double value);  //jni
	public native void setBool(boolean value);   //jni
	public native void setString(String value);  //jni

	public native void setDataPtr(byte[] value); //jni

	public native void pushBack_jni(Context data, boolean copy);

	public native long size();           //jni

	public native boolean isNone();      //jni
	public native boolean isArray();     //jni
	public native boolean isObject();    //jni
	public native boolean isBool();      //jni
	public native boolean isLong();      //jni
	public native boolean isDouble();    //jni
	public native boolean isString();    //jni
	public native boolean isDataPtr();   //jni

	public native long getLong();        //jni
	public native double getDouble();    //jni
	public native boolean getBool();     //jni
	public native String getString();    //jni
	public native byte [] getDataPtr();  //jni
}

