# 🌐 **How to Test Your HTML Interface**

## ✅ **Fixed Issue**

**Problem**: The HTML file was pointing to `http://localhost:8001/api/v1` but your server is running on port `8000`.

**Solution**: Updated `test_interface.html` to use `http://localhost:8000/api/v1`.

---

## 🚀 **Step-by-Step Testing**

### **1. Make Sure Server is Running**
```bash
cd /Users/alan/yaadrolens1.1/face_recognition_attendance
source venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

**You should see:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000
Database tables created successfully
INFO:     Application startup complete.
```

### **2. Test API Connection**
Open `test_html_connection.html` in your browser first to verify the connection:
- Double-click `test_html_connection.html`
- You should see: "✅ Connection successful!"

### **3. Open Main Interface**
Open `test_interface.html` in your browser:
- Double-click `test_interface.html`
- Or drag it into your browser window

### **4. Test Basic Functionality**

**Login First:**
1. Click the "Login" button in the header
2. Use default credentials:
   - Email: `admin@test.com`
   - Password: `admin123`

**Test Employee Enrollment:**
1. Go to "Employee Enrollment" tab
2. Fill in employee details
3. Click "Start Camera" 
4. Click "🤖 Start Auto-Capture"
5. It should capture 20-30 images automatically

**Test Attendance:**
1. Go to "Clock In/Out" tab
2. Click "Start Camera"
3. Click "Clock In/Out" when face is detected

---

## 🔧 **Troubleshooting**

### **Common Issues:**

**1. "Connection failed" in browser:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Should return: {"status":"healthy","message":"API is running"}
```

**2. "CORS error" in browser console:**
- The server has CORS enabled for all origins
- Make sure you're opening the HTML file in a browser (not just viewing the code)

**3. "Camera not working":**
- Grant camera permissions when prompted
- Make sure no other apps are using the camera

**4. "No faces detected":**
- Ensure good lighting
- Position face clearly in camera view
- Add photos to `dataset/person_name/` folders first

### **Debug Steps:**

**Check Server Status:**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/auth/me
```

**Check Browser Console:**
- Press F12 in browser
- Look for any error messages in Console tab

**Check Network Tab:**
- Press F12 → Network tab
- Try an action in the interface
- See if API calls are being made

---

## 📸 **Before Testing Recognition**

**Add sample photos:**
```bash
mkdir -p dataset/test_person
# Add 5-10 photos to dataset/test_person/
# Name them: 1.jpg, 2.jpg, 3.jpg, etc.
```

**Photo requirements:**
- Clear face visible
- Good lighting
- Different angles
- JPG/PNG format

---

## 🎯 **Expected Results**

### **Working Interface Should Show:**
- ✅ "DeepFace + ArcFace System Active" status
- ✅ Camera preview when started
- ✅ Auto-capture progress bar during enrollment
- ✅ Face quality indicator during attendance
- ✅ Real-time recognition results

### **API Endpoints Working:**
- `GET /health` → `{"status":"healthy"}`
- `POST /api/v1/auth/login` → Returns access token
- `POST /api/v1/employees/register_face` → Saves images
- `POST /api/v1/attendance/clock` → Processes attendance

---

## 🎉 **You're Ready!**

Your HTML interface should now work perfectly with:
- 🌐 **Correct API endpoint** (port 8000)
- 🚀 **DeepFace + ArcFace system** 
- 📸 **Auto-capture enrollment**
- ⏰ **Real-time attendance tracking**
- 🔒 **Face quality validation**

**Start testing:** Open `test_interface.html` in your browser!

---

## 📞 **Still Having Issues?**

1. **Test connection first:** Open `test_html_connection.html`
2. **Check server logs** in terminal for errors
3. **Check browser console** (F12) for JavaScript errors
4. **Verify photos** are in `dataset/person_name/` folders

**Your system is ready to go! 🚀✨**

