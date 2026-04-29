# Aisha Privacy Policy

**Effective date:** April 24, 2026

Aisha ("the app") is an on-device conversational application
published by Sauli Kälkäjä.  This policy describes what the app
does with data.

## 1. Personal data we collect

None.  The app does not collect, store, or transmit any personal
data about you.  We have no accounts, no sign-in, and no way to
identify individual users.

## 2. Conversation data

Your conversations with Aisha are stored only on your own device,
in the app's private storage.  We cannot see them.  They are not
sent to any server.

## 3. Network usage

The app makes outbound network requests only in two situations:

**a) First launch.**  Aisha downloads the manifold data files
(approximately 165 MB total) from Hugging Face
(`huggingface.co`) so the engine can run offline afterwards.  This
download happens once per install.  No personal data is sent — it's
a plain `GET` for public files.

**b) Factual questions.**  When you explicitly ask a question that
the app routes as a factual lookup (e.g. "when did X die?"), Aisha
queries Wikipedia (`wikipedia.org`).  Only the query text you typed
is transmitted; no identifier, no device ID, no account.

## 4. Advertising and tracking

The app contains no advertising networks, no analytics SDKs, no
crash reporting that leaves the device, and no third-party
trackers.

## 5. Permissions

The app requests the `INTERNET` permission for the two cases in §3.
No other runtime permissions.

## 6. Children

The app is not directed at children under 13 and does not knowingly
collect data from anyone.  (There is no data to collect.)

## 7. Changes to this policy

If this policy changes, the updated version will appear at this same
URL with a new effective date.

## 8. Contact

For questions about this policy, open an issue at:
[github.com/SauliKalkaja/Aisha-code](https://github.com/SauliKalkaja/Aisha-code)
