package com.verba.app;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.speech.tts.TextToSpeech;
import android.speech.tts.UtteranceProgressListener;
import android.webkit.JavascriptInterface;
import android.webkit.PermissionRequest;
import android.webkit.WebSettings;
import android.webkit.WebView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import com.getcapacitor.BridgeActivity;
import com.getcapacitor.BridgeWebChromeClient;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Locale;

public class MainActivity extends BridgeActivity {
    private static final int PERMISSION_REQUEST_CODE = 1001;
    private VerbaNativeBridge nativeBridge;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 1. Proactively request runtime camera & audio permissions if not granted
        checkAndRequestPermissions();

        // 2. Configure WebView settings
        WebView webView = this.bridge != null ? this.bridge.getWebView() : null;
        if (webView != null) {
            WebSettings settings = webView.getSettings();
            // Allow programmatic video playback without user gesture for SignPlayer continuous playback
            settings.setMediaPlaybackRequiresUserGesture(false);
            settings.setDomStorageEnabled(true);
            settings.setAllowFileAccess(true);
            settings.setAllowContentAccess(true);
            settings.setDatabaseEnabled(true);

            // 3. Set custom WebChromeClient to reliably grant camera & audio permissions to WebView
            webView.setWebChromeClient(new BridgeWebChromeClient(this.bridge) {
                @Override
                public void onPermissionRequest(final PermissionRequest request) {
                    runOnUiThread(() -> {
                        boolean hasCamera = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
                        boolean hasMic = ContextCompat.checkSelfPermission(MainActivity.this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED;
                        
                        if (hasCamera && hasMic) {
                            request.grant(request.getResources());
                        } else {
                            // Request through base client which prompts the user
                            super.onPermissionRequest(request);
                        }
                    });
                }
            });

            // 4. Register native bridge for Speech Synthesis (TTS) & Speech Recognition (STT)
            nativeBridge = new VerbaNativeBridge(this, webView);
            webView.addJavascriptInterface(nativeBridge, "VerbaNative");
        }
    }

    private void checkAndRequestPermissions() {
        String[] permissions = new String[]{
            Manifest.permission.CAMERA,
            Manifest.permission.RECORD_AUDIO,
            Manifest.permission.MODIFY_AUDIO_SETTINGS
        };
        ArrayList<String> toRequest = new ArrayList<>();
        for (String perm : permissions) {
            if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                toRequest.add(perm);
            }
        }
        if (!toRequest.isEmpty()) {
            ActivityCompat.requestPermissions(this, toRequest.toArray(new String[0]), PERMISSION_REQUEST_CODE);
        }
    }

    @Override
    public void onDestroy() {
        if (nativeBridge != null) {
            nativeBridge.destroy();
        }
        super.onDestroy();
    }

    public static class VerbaNativeBridge {
        private final MainActivity activity;
        private final WebView webView;
        private final Handler mainHandler;
        private TextToSpeech tts;
        private SpeechRecognizer speechRecognizer;
        private boolean ttsInitialized = false;

        public VerbaNativeBridge(MainActivity activity, WebView webView) {
            this.activity = activity;
            this.webView = webView;
            this.mainHandler = new Handler(Looper.getMainLooper());
            initTTS();
        }

        private void initTTS() {
            mainHandler.post(() -> {
                tts = new TextToSpeech(activity.getApplicationContext(), status -> {
                    if (status == TextToSpeech.SUCCESS) {
                        ttsInitialized = true;
                        tts.setLanguage(new Locale("en", "IN"));
                        tts.setSpeechRate(0.95f);
                        tts.setOnUtteranceProgressListener(new UtteranceProgressListener() {
                            @Override
                            public void onStart(String utteranceId) {
                                notifyTTSStatus(true);
                            }

                            @Override
                            public void onDone(String utteranceId) {
                                notifyTTSStatus(false);
                            }

                            @Override
                            public void onError(String utteranceId) {
                                notifyTTSStatus(false);
                            }
                        });
                    }
                });
            });
        }

        private void notifyTTSStatus(boolean isSpeaking) {
            mainHandler.post(() -> {
                if (webView != null) {
                    webView.evaluateJavascript("window.onVerbaTTSStatus && window.onVerbaTTSStatus(" + isSpeaking + ");", null);
                }
            });
        }

        @JavascriptInterface
        public boolean isNative() {
            return true;
        }

        @JavascriptInterface
        public void speak(String text, String lang) {
            if (text == null || text.trim().isEmpty()) return;
            mainHandler.post(() -> {
                if (tts != null && ttsInitialized) {
                    Locale loc = "hi".equalsIgnoreCase(lang) ? new Locale("hi", "IN") : new Locale("en", "IN");
                    int langResult = tts.setLanguage(loc);
                    if (langResult == TextToSpeech.LANG_MISSING_DATA || langResult == TextToSpeech.LANG_NOT_SUPPORTED) {
                        tts.setLanguage(Locale.US);
                    }
                    tts.speak(text, TextToSpeech.QUEUE_FLUSH, null, "verba_tts_" + System.currentTimeMillis());
                }
            });
        }

        @JavascriptInterface
        public void stopSpeaking() {
            mainHandler.post(() -> {
                if (tts != null) {
                    tts.stop();
                }
                notifyTTSStatus(false);
            });
        }

        @JavascriptInterface
        public void startListening(String lang) {
            mainHandler.post(() -> {
                try {
                    if (speechRecognizer != null) {
                        speechRecognizer.destroy();
                    }
                    if (!SpeechRecognizer.isRecognitionAvailable(activity)) {
                        notifySpeechError("Speech recognition not available on device");
                        return;
                    }
                    speechRecognizer = SpeechRecognizer.createSpeechRecognizer(activity);
                    speechRecognizer.setRecognitionListener(new RecognitionListener() {
                        @Override
                        public void onReadyForSpeech(Bundle params) {}

                        @Override
                        public void onBeginningOfSpeech() {}

                        @Override
                        public void onRmsChanged(float rmsdB) {}

                        @Override
                        public void onBufferReceived(byte[] buffer) {}

                        @Override
                        public void onEndOfSpeech() {
                            notifySpeechEnd();
                        }

                        @Override
                        public void onError(int error) {
                            String message;
                            switch (error) {
                                case SpeechRecognizer.ERROR_AUDIO: message = "Audio recording error"; break;
                                case SpeechRecognizer.ERROR_CLIENT: message = "Client error"; break;
                                case SpeechRecognizer.ERROR_INSUFFICIENT_PERMISSIONS: message = "Microphone permission required"; break;
                                case SpeechRecognizer.ERROR_NETWORK: message = "Network error"; break;
                                case SpeechRecognizer.ERROR_NETWORK_TIMEOUT: message = "Network timeout"; break;
                                case SpeechRecognizer.ERROR_NO_MATCH: message = "No speech recognized"; break;
                                case SpeechRecognizer.ERROR_RECOGNIZER_BUSY: message = "Recognition service busy"; break;
                                case SpeechRecognizer.ERROR_SERVER: message = "Server error"; break;
                                case SpeechRecognizer.ERROR_SPEECH_TIMEOUT: message = "Speech timeout"; break;
                                default: message = "Speech error " + error; break;
                            }
                            notifySpeechError(message);
                        }

                        @Override
                        public void onResults(Bundle results) {
                            ArrayList<String> matches = results.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                            if (matches != null && !matches.isEmpty()) {
                                String text = matches.get(0);
                                notifySpeechResult(text, true);
                            } else {
                                notifySpeechEnd();
                            }
                        }

                        @Override
                        public void onPartialResults(Bundle partialResults) {
                            ArrayList<String> matches = partialResults.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);
                            if (matches != null && !matches.isEmpty()) {
                                String text = matches.get(0);
                                notifySpeechResult(text, false);
                            }
                        }

                        @Override
                        public void onEvent(int eventType, Bundle params) {}
                    });

                    Intent intent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
                    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
                    intent.putExtra(RecognizerIntent.EXTRA_LANGUAGE, "hi".equalsIgnoreCase(lang) ? "hi-IN" : "en-IN");
                    intent.putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true);
                    speechRecognizer.startListening(intent);
                } catch (Exception e) {
                    notifySpeechError(e.getMessage());
                }
            });
        }

        @JavascriptInterface
        public void stopListening() {
            mainHandler.post(() -> {
                if (speechRecognizer != null) {
                    try {
                        speechRecognizer.stopListening();
                    } catch (Exception ignored) {}
                }
            });
        }

        private void notifySpeechResult(String text, boolean isFinal) {
            mainHandler.post(() -> {
                if (webView != null) {
                    String escaped = text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ");
                    webView.evaluateJavascript("window.onVerbaSpeechResult && window.onVerbaSpeechResult('" + escaped + "', " + isFinal + ");", null);
                }
            });
        }

        private void notifySpeechError(String error) {
            mainHandler.post(() -> {
                if (webView != null) {
                    String escaped = error.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ");
                    webView.evaluateJavascript("window.onVerbaSpeechError && window.onVerbaSpeechError('" + escaped + "');", null);
                }
            });
        }

        private void notifySpeechEnd() {
            mainHandler.post(() -> {
                if (webView != null) {
                    webView.evaluateJavascript("window.onVerbaSpeechEnd && window.onVerbaSpeechEnd();", null);
                }
            });
        }

        public void destroy() {
            mainHandler.post(() -> {
                if (tts != null) {
                    tts.stop();
                    tts.shutdown();
                    tts = null;
                }
                if (speechRecognizer != null) {
                    speechRecognizer.destroy();
                    speechRecognizer = null;
                }
            });
        }
    }
}
