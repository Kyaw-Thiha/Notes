# Bluetooth

Quick reference for:
1. **Finding / pairing / connecting** Bluetooth devices
2. **Switching audio profiles** (music vs mic) with `pavucontrol`

---

## 1. Finding, pairing, connecting

### 1.1 Start `bluetoothctl` interactive

```bash
bluetoothctl
```

Inside the prompt:

```text
agent on
default-agent
power on
pairable on
```

### 1.2 Scan for devices

```text
scan on
```

- Put device (headphones, etc.) into **pairing mode**.
- Watch for a line like:

```text
[NEW] Device AA:BB:CC:DD:EE:FF Device-Name
```

(That MAC address is what you’ll use below.)

To stop scanning:

```text
scan off
```

### 1.3 Pair, trust, connect

```text
pair AA:BB:CC:DD:EE:FF
trust AA:BB:CC:DD:EE:FF
connect AA:BB:CC:DD:EE:FF
```

If successful you’ll see `Connected: yes`.

### 1.4 List devices / paired devices

```text
devices
paired-devices
info AA:BB:CC:DD:EE:FF
```

### 1.5 Remove a broken device

If you get key errors, or connection is cursed:

```text
remove AA:BB:CC:DD:EE:FF
```

Then re-do: `scan on` → `pair` → `trust` → `connect`.

### 1.6 Quit bluetoothctl

```text
quit
```

---

## 2. Audio profiles with `pavucontrol` (music vs mic)

### 2.1 Open `pavucontrol`

Install once (if needed):

```bash
sudo pacman -S pavucontrol
```

Run:

```bash
pavucontrol
```

---

### 2.2 Switch between “music mode” and “mic mode”

In `pavucontrol`:

#### Configuration tab

- Find your **Bluetooth headset**.
- Choose **one**:

1. **High Fidelity Playback (A2DP Sink)**  
   - Good sound, **no mic**.
2. **Headset Head Unit (HSP/HFP)**  
   - Mic works, **low audio quality** (phone call sound).

Pick depending on what you’re doing.

---

### 2.3 Set active output / input

#### Output Devices tab

- Select your **headphones**.
- Optionally click the green checkmark to make them **default output**.

#### Input Devices tab

- For headset mic: select the **Bluetooth headset** and unmute it.
- For “nice audio + laptop mic” setup:
  - Keep headset on `High Fidelity (A2DP)` in **Configuration**.
  - Use **internal laptop mic** as the input device.

---

### 2.4 Discord-specific

In Discord → **Settings → Voice & Video**:

- **Output Device**: Bluetooth headphones.
- **Input Device**:
  - Use headset device if you chose `Headset Head Unit`.
  - Use `Internal Microphone` (or USB mic) if staying on `A2DP` for better audio.

Use Mic Test in Discord to confirm the bar moves when talking.

---

