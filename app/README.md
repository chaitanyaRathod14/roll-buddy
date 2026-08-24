# Roll Call — Android App (Capacitor wrapper)

Your existing website (`www/index.html`) wrapped into a real installable Android
app using Capacitor. No logic was changed — same DeepFace/MongoDB backend
(`app.py`), same face detection/recognition, same enrollment flow. Capacitor
just puts your website inside a native Android shell so it installs and runs
like an app, with real camera access (the browser's `getUserMedia` API works
fine inside Capacitor's WebView — no extra camera plugin needed).

## Before you start: decide your backend URL

Open `www/index.html`, find this line near the top of the `<script>` block:

```js
const API_BASE = "http://localhost:5001";
```

Change it based on how you're testing:

| Testing method | Set API_BASE to |
|---|---|
| Browser, same laptop as `app.py` | `http://localhost:5001` |
| **Android emulator** (recommended — no network issues) | `http://10.0.2.2:5001` |
| Physical Android phone, same WiFi as laptop | `http://<your-laptop-LAN-IP>:5001` |

**Strong recommendation: use the Android emulator.** `10.0.2.2` is a special
alias Android emulators use to reach `localhost` on the host machine — this
completely avoids WiFi/firewall/hotspot issues since everything runs on one
machine.

## Setup

1. Make sure your `app.py` backend and MongoDB are running first (same as
   before — nothing about them changes).

2. Install Node.js dependencies for the wrapper:
   ```bash
   npm install
   ```

3. Install the Capacitor CLI globally (if not already):
   ```bash
   npm install -g @capacitor/cli
   ```

4. Initialize the Android project:
   ```bash
   npx cap add android
   ```

5. Sync your website into the Android project:
   ```bash
   npx cap sync android
   ```

6. Open it in Android Studio:
   ```bash
   npx cap open android
   ```

7. In Android Studio: create/select a virtual device (Device Manager → Play
   button), then click the green **Run** button. The app installs and opens
   on the emulator automatically.

## Camera permission

Capacitor's default Android template already includes camera permission in
`android/app/src/main/AndroidManifest.xml`. If the app can't access the
camera when you tap "Enroll" or "Take attendance", open that file and confirm
this line is present:

```xml
<uses-permission android:name="android.permission.CAMERA" />
```

If missing, add it inside the `<manifest>` tag, then re-run `npx cap sync android`.

## Every time you edit index.html

Whenever you change `www/index.html`, re-sync before testing again:
```bash
npx cap sync android
```
Then re-run from Android Studio.

## Building a final APK for submission

Once everything works in the emulator:
1. In Android Studio: **Build → Build Bundle(s) / APK(s) → Build APK(s)**
2. Find the generated `.apk` under `android/app/build/outputs/apk/debug/`
3. This is your installable, submittable Android app

## If you switch to a physical phone later

Change `API_BASE` in `index.html` to your laptop's LAN IP, re-run
`npx cap sync android`, and make sure your phone and laptop are on the same
WiFi with any firewall/antivirus (e.g. Avast) allowing the connection —
same networking rules as before, just now wrapping a website instead of a
React Native app.
