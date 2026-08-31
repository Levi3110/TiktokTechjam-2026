#!/usr/bin/env python3
"""Local web UI for the TechJam conversational shopping agent."""

from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import parse_qs, urlparse

from livekit import api

from starter.agent import Agent
from starter.behavior import BehaviorStore, ProductImageCache


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>NAmazon · Your Shopping Companion</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Inter, ui-rounded, "SF Pro Rounded", ui-sans-serif, system-ui, sans-serif;
      --ink: #342c45;
      --muted: #81778e;
      --primary: #7457a6;
      --primary-dark: #5d428f;
      --pink: #f38cac;
      --peach: #ffddcf;
      --mint: #d6f1e8;
      --surface: rgba(255, 255, 255, .9);
      --line: #eadff0;
      --shadow: 0 20px 55px rgba(92, 66, 126, .13);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background:
        radial-gradient(circle at 8% 4%, rgba(255, 203, 219, .75), transparent 28%),
        radial-gradient(circle at 91% 14%, rgba(208, 238, 233, .9), transparent 27%),
        linear-gradient(145deg, #fffafc 0%, #f7f3ff 52%, #fff8f1 100%);
    }
    body::before, body::after {
      content: "";
      position: fixed;
      width: 170px;
      height: 170px;
      border-radius: 42% 58% 64% 36%;
      filter: blur(1px);
      opacity: .32;
      pointer-events: none;
      z-index: -1;
    }
    body::before { top: 18%; left: -80px; background: #e4d5fa; transform: rotate(25deg); }
    body::after { right: -75px; bottom: 8%; background: #ffd4df; transform: rotate(-18deg); }
    main { width: min(1180px, calc(100% - 36px)); margin: 24px auto 34px; }
    header { display: flex; justify-content: space-between; align-items: center; gap: 18px; margin-bottom: 20px; }
    .brand { display: flex; align-items: center; gap: 13px; }
    .brand-mark { display: grid; place-items: center; width: 52px; height: 52px; border-radius: 18px; background: linear-gradient(145deg, #8c6fc0, #ef91b1); color: #fff; font-size: 25px; box-shadow: 0 12px 25px rgba(116,87,166,.22); }
    .eyebrow { margin: 0 0 3px; color: var(--primary); font-size: 11px; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(25px, 4vw, 38px); letter-spacing: -.04em; }
    .sub { color: var(--muted); margin: 5px 0 0; font-size: 14px; }
    button { border: 0; border-radius: 14px; padding: 11px 16px; background: #f1ebf7; color: var(--ink); font: inherit; font-weight: 700; cursor: pointer; transition: transform .18s ease, box-shadow .18s ease, background .18s ease; }
    button:hover { transform: translateY(-2px); box-shadow: 0 9px 20px rgba(91,66,125,.12); }
    button:focus-visible { outline: 3px solid rgba(116,87,166,.25); outline-offset: 2px; }
    button:disabled { opacity: .5; cursor: wait; transform: none; box-shadow: none; }
    .header-actions { display: flex; gap: 9px; }
    #avatar-connect { background: #fff; border: 1px solid var(--line); }
    #reset { background: var(--primary); color: #fff; }
    .flow-picker { margin-bottom: 18px; padding: 20px; border: 1px solid rgba(222,207,233,.9); border-radius: 24px; background: var(--surface); box-shadow: var(--shadow); backdrop-filter: blur(18px); }
    .flow-picker h2 { margin: 0 0 6px; font-size: 21px; letter-spacing: -.02em; }
    .flow-picker p { margin: 0 0 16px; color: var(--muted); font-size: 13px; }
    .flow-options { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .flow-option { display: grid; grid-template-columns: 46px 1fr; column-gap: 12px; padding: 16px; text-align: left; border: 1px solid var(--line); background: #fffafd; }
    .flow-option:last-child { background: #f8fffc; }
    .flow-option:hover { border-color: #cdb8df; background: #fff; }
    .flow-icon { grid-row: 1 / 3; display: grid; place-items: center; width: 46px; height: 46px; border-radius: 15px; background: #f7deea; font-size: 22px; }
    .flow-option:last-child .flow-icon { background: var(--mint); }
    .flow-option strong, .flow-option span { display: block; }
    .flow-option strong { align-self: end; margin-bottom: 3px; font-size: 15px; }
    .flow-option span { color: var(--muted); font-size: 12px; font-weight: 500; line-height: 1.35; }
    .workspace { display: grid; grid-template-columns: 350px minmax(0, 1fr); gap: 18px; align-items: stretch; }
    .avatar-panel, .conversation { border: 1px solid rgba(226,213,234,.92); border-radius: 26px; background: var(--surface); box-shadow: var(--shadow); backdrop-filter: blur(18px); }
    .avatar-panel { padding: 14px; }
    .avatar-stage { position: relative; height: 490px; overflow: hidden; border-radius: 20px; background: linear-gradient(155deg, #eee5fa, #fce7ef 55%, #e5f5f0); display: grid; place-items: center; }
    .avatar-stage::after { content: "✦"; position: absolute; top: 18px; right: 20px; color: rgba(116,87,166,.42); font-size: 24px; z-index: 3; }
    .avatar-stage video { display: none; }
    .face-fallback { position: relative; z-index: 2; width: 180px; height: 238px; border-radius: 50% 50% 45% 45%; background: linear-gradient(155deg, #f2c7aa 4%, #d6a083 58%, #c78f77); box-shadow: inset 16px 12px 32px rgba(255,255,255,.2), inset -14px -16px 28px rgba(120,72,77,.08), 0 28px 60px rgba(89,57,74,.2); animation: eggFloat 3.6s ease-in-out infinite; }
    .face-fallback::before, .face-fallback::after { content: ""; position: absolute; top: 91px; width: 15px; height: 9px; border-radius: 50%; background: #493d55; animation: blink 4.8s ease-in-out infinite; }
    .face-fallback::before { left: 47px; } .face-fallback::after { right: 47px; }
    .mouth { position: absolute; left: 72px; top: 158px; width: 36px; height: 9px; border-radius: 4px 4px 26px 26px; background: #914d61; transform-origin: 50% 0; transition: height .055s linear, width .055s linear, left .055s linear, border-radius .055s linear; box-shadow: inset 0 3px 5px rgba(72,34,55,.14); }
    .avatar-stage.speaking .face-fallback { box-shadow: inset 16px 12px 32px rgba(255,255,255,.2), inset -14px -16px 28px rgba(120,72,77,.08), 0 26px 66px rgba(150,100,170,.3); }
    .avatar-halo { position: absolute; width: 270px; height: 270px; border-radius: 50%; border: 1px solid rgba(151,116,190,.3); box-shadow: 0 0 75px rgba(151,116,190,.18); animation: breathe 2.5s ease-in-out infinite; }
    .avatar-badge { position: absolute; z-index: 4; left: 12px; right: 12px; bottom: 12px; display: flex; align-items: center; gap: 9px; padding: 11px 13px; border: 1px solid rgba(255,255,255,.72); border-radius: 15px; background: rgba(255,255,255,.82); box-shadow: 0 9px 24px rgba(66,47,82,.12); color: #51465f; font-size: 12px; font-weight: 700; backdrop-filter: blur(12px); }
    .dot { width: 8px; height: 8px; flex: 0 0 auto; border-radius: 50%; background: #e9a75c; box-shadow: 0 0 0 4px rgba(233,167,92,.14); }
    .dot.online { background: #55b990; box-shadow: 0 0 0 4px rgba(85,185,144,.14); }
    .avatar-note { color: var(--muted); font-size: 12px; line-height: 1.5; margin: 11px 5px 1px; text-align: center; }
    .conversation { min-width: 0; padding: 17px; display: flex; flex-direction: column; }
    #chat { height: min(560px, calc(100vh - 220px)); min-height: 450px; overflow: auto; padding: 5px 5px 5px 1px; scrollbar-color: #d8c8e4 transparent; }
    .bubble { max-width: 82%; margin: 0 0 14px; padding: 13px 16px; border-radius: 18px; line-height: 1.5; white-space: pre-wrap; box-shadow: 0 7px 18px rgba(91,66,125,.07); }
    .user { margin-left: auto; border-bottom-right-radius: 6px; background: linear-gradient(145deg, #8064b2, #6d50a0); color: #fff; }
    .agent { border: 1px solid #eee3f2; border-bottom-left-radius: 6px; background: #fff; }
    .products { display: grid; gap: 9px; margin-top: 13px; }
    .product { padding: 12px 13px; background: linear-gradient(135deg, #fff9fc, #faf7ff); border: 1px solid #eadff0; border-radius: 14px; }
    .product strong { display: block; font-size: 14px; }
    .product span { color: var(--muted); font-size: 12px; }
    .choose-product { margin-top: 9px; padding: 8px 11px; border: 1px solid #dfd0e9; background: #fff; color: var(--primary-dark); font-size: 12px; }
    .choice-actions { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 11px; }
    .choice-actions button { padding: 8px 12px; border: 1px solid #dfd0e9; background: #fff; color: var(--primary-dark); font-size: 12px; }
    .choice-actions .confirm-choice { border-color: transparent; background: var(--primary); color: #fff; }
    .confirmed-product { display: grid; grid-template-columns: 92px 1fr; gap: 12px; align-items: center; margin-top: 12px; padding: 12px; border: 1px solid #dfd0e9; border-radius: 16px; background: linear-gradient(135deg, #fff9fc, #f4faff); }
    .confirmed-product img, .product-image-placeholder { width: 92px; height: 92px; border-radius: 13px; object-fit: contain; background: #fff; }
    .product-image-placeholder { display: grid; place-items: center; color: var(--primary); font-size: 28px; }
    .confirmed-product strong { display: block; margin-bottom: 5px; font-size: 13px; }
    .confirmed-product span { color: var(--muted); font-size: 12px; }
    form { display: flex; gap: 9px; margin-top: auto; padding-top: 12px; }
    input { flex: 1; min-width: 0; border: 1px solid #dfd2e8; border-radius: 16px; background: #fff; color: var(--ink); padding: 15px 16px; font: inherit; font-size: 15px; outline: none; }
    input::placeholder { color: #aaa0b4; }
    input:focus { border-color: #a58ac5; box-shadow: 0 0 0 4px rgba(116,87,166,.1); }
    #mic { width: 52px; flex: 0 0 52px; padding: 0; font-size: 20px; background: #f7dce7; }
    #mic.listening { background: var(--pink); color: #fff; animation: micPulse 1s infinite; }
    #send { padding-inline: 21px; background: var(--primary); color: #fff; }
    #send:hover { background: var(--primary-dark); }
    .voice-controls { display: flex; align-items: center; gap: 8px; padding: 8px 3px 0; color: var(--muted); font-size: 12px; }
    .voice-controls[hidden] { display: none; }
    .voice-controls select { min-width: 0; max-width: 260px; border: 1px solid #dfd2e8; border-radius: 10px; background: #fff; color: var(--ink); padding: 7px 10px; font: inherit; }
    #status { min-height: 25px; padding: 9px 3px 0; color: var(--muted); font-size: 12px; }
    #status::before { content: "•"; margin-right: 6px; color: var(--pink); }
    @keyframes eggFloat { 50% { transform: translateY(-7px); } }
    @keyframes blink { 0%, 46%, 50%, 100% { transform: scaleY(1); } 48% { transform: scaleY(.12); } }
    @keyframes breathe { 50% { transform: scale(1.05); opacity: .7; } }
    @keyframes micPulse { 50% { box-shadow: 0 0 0 9px rgba(243,140,172,.16); } }
    @media (max-width: 800px) { .workspace { grid-template-columns: 1fr; } .avatar-stage { height: 380px; } #chat { height: auto; min-height: 50vh; } }
    @media (max-width: 620px) { .flow-options { grid-template-columns: 1fr; } }
    @media (max-width: 600px) { main { width: min(100% - 22px, 1180px); margin: 12px auto 22px; } header { align-items: flex-start; } .brand-mark { width: 45px; height: 45px; } .header-actions { flex-direction: column; } .header-actions button { padding: 9px 12px; font-size: 12px; } .bubble { max-width: 94%; } .flow-picker { padding: 16px; } }
  </style>
</head>
<body>
<main>
  <header>
    <div class="brand">
      <div class="brand-mark">✦</div>
      <div><p class="eyebrow">Your personal shopping companion</p><h1>Meet NAmazon</h1><p class="sub">A thoughtful little helper for finding things you will love.</p></div>
    </div>
    <div class="header-actions">
      <button id="avatar-connect" type="button">Reconnect assistant</button>
      <button id="reset" type="button">New chat</button>
    </div>
  </header>
  <section class="flow-picker" id="flow-picker">
    <h2>How can I help today?</h2>
    <p>Pick a starting point. You can always change direction as we chat.</p>
    <div class="flow-options">
      <button class="flow-option" type="button" data-flow="buying">
        <span class="flow-icon">🛍️</span>
        <strong>Shop with a goal</strong>
        <span>Tell me what you need and I will help narrow it down.</span>
      </button>
      <button class="flow-option" type="button" data-flow="browsing">
        <span class="flow-icon">✨</span>
        <strong>Browse for inspiration</strong>
        <span>Explore lovely ideas, categories and styles together.</span>
      </button>
    </div>
  </section>
  <div class="workspace">
    <aside class="avatar-panel">
      <div class="avatar-stage" id="avatar-stage">
        <div class="avatar-halo"></div>
        <div class="face-fallback"><div class="mouth"></div></div>
        <video id="avatar-video" autoplay playsinline muted></video>
        <audio id="avatar-audio" autoplay></audio>
        <div class="avatar-badge"><span class="dot" id="avatar-dot"></span><span id="avatar-status">Assistant is getting ready</span></div>
      </div>
      <input id="livetalking-url" type="hidden" value="http://127.0.0.1:8010">
      <p class="avatar-note">Type a message or tap the microphone to chat naturally.</p>
    </aside>
    <section class="conversation">
      <section id="chat" aria-live="polite">
        <div class="bubble agent">Hi, I’m NAmazon! Tell me what you are looking for, or let’s discover something delightful together.</div>
      </section>
      <form id="form">
        <button id="mic" type="button" title="Start a voice message" aria-label="Start a voice message" disabled>🎙</button>
        <input id="message" autocomplete="off" placeholder="Choose an option above to begin..." disabled autofocus>
        <button id="send" type="submit" disabled>Send</button>
      </form>
      <div class="voice-controls" id="voice-controls" hidden>
        <label for="mic-device">Microphone</label>
        <select id="mic-device" aria-label="Microphone device"></select>
      </div>
      <div id="status">Ready when you are</div>
    </section>
  </div>
</main>
<script src="/static/vendor/livekit-client.umd.min.js"></script>
<script>
  const chat = document.querySelector('#chat');
  const form = document.querySelector('#form');
  const input = document.querySelector('#message');
  const send = document.querySelector('#send');
  const status = document.querySelector('#status');
  const mic = document.querySelector('#mic');
  const micDevice = document.querySelector('#mic-device');
  const voiceControls = document.querySelector('#voice-controls');
  const avatarStage = document.querySelector('#avatar-stage');
  const eggMouth = document.querySelector('.mouth');
  const avatarVideo = document.querySelector('#avatar-video');
  const avatarAudio = document.querySelector('#avatar-audio');
  const avatarStatus = document.querySelector('#avatar-status');
  const avatarDot = document.querySelector('#avatar-dot');
  const avatarConnect = document.querySelector('#avatar-connect');
  const liveUrl = document.querySelector('#livetalking-url');
  const flowPicker = document.querySelector('#flow-picker');
  let sessionId = crypto.randomUUID();
  let clientId;
  try {
    clientId = localStorage.getItem('namazon-client-id') || crypto.randomUUID();
    localStorage.setItem('namazon-client-id', clientId);
  } catch (_) { clientId = crypto.randomUUID(); }
  let turn = 1;
  let selectedFlow = null;
  let avatarPC = null;
  let avatarSessionId = null;
  let livekitRoom = null;
  let livekitTrack = null;
  let livekitConnecting = null;
  let micPermissionReady = false;
  let audioContext = null;
  let remoteAudioSource = null;
  let lipAnalyser = null;
  let lipAnimationFrame = 0;
  let smoothedMouthLevel = 0;
  let fallbackMouthTimer = null;

  function bubble(role, text, payload) {
    const box = document.createElement('div');
    box.className = `bubble ${role}`;
    const copy = document.createElement('div');
    copy.textContent = text;
    box.appendChild(copy);
    if (payload?.recommendations?.length) {
      const list = document.createElement('div');
      list.className = 'products';
      payload.recommendations.forEach((item, index) => {
        const row = document.createElement('div');
        row.className = 'product';
        const title = document.createElement('strong');
        title.textContent = `${index + 1}. ${item.title || item.parent_asin}`;
        const detail = document.createElement('span');
        detail.textContent = item.price == null ? 'Recommended for you' : `$${item.price}`;
        const choose = document.createElement('button');
        choose.type = 'button'; choose.className = 'choose-product';
        choose.textContent = 'Choose this';
        choose.addEventListener('click', () => beginProductChoice(item));
        row.append(title, detail, choose); list.appendChild(row);
      });
      box.appendChild(list);
    }
    if (payload?.confirmed_product) {
      const item = payload.confirmed_product;
      const card = document.createElement('div'); card.className = 'confirmed-product';
      if (item.image_url) {
        const image = document.createElement('img');
        image.src = item.image_url; image.alt = item.title || 'Confirmed product';
        card.appendChild(image);
      } else {
        const placeholder = document.createElement('div');
        placeholder.className = 'product-image-placeholder'; placeholder.textContent = '✦';
        card.appendChild(placeholder);
      }
      const details = document.createElement('div');
      const title = document.createElement('strong'); title.textContent = item.title;
      const selectedSize = document.createElement('span'); selectedSize.textContent = `Selected size: ${item.size}`;
      details.append(title, selectedSize); card.appendChild(details); box.appendChild(card);
    }
    chat.appendChild(box); chat.scrollTop = chat.scrollHeight;
    return box;
  }

  async function selectionRequest(action, item, size = '') {
    const response = await fetch('/api/selection', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        session_id: sessionId,
        client_id: clientId,
        parent_asin: item.parent_asin,
        action,
        size,
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
    return payload;
  }

  async function beginProductChoice(item) {
    try {
      await selectionRequest('select', item);
      const box = bubble('agent', `Great choice — you’re choosing ${item.title}. Which size would you like?`);
      const actions = document.createElement('div'); actions.className = 'choice-actions';
      ['XS', 'S', 'M', 'L', 'XL', 'One size'].forEach(size => {
        const button = document.createElement('button'); button.type = 'button'; button.textContent = size;
        button.addEventListener('click', async () => {
          try {
            await selectionRequest('size', item, size);
            actions.replaceChildren();
            const summary = document.createElement('span');
            summary.textContent = `Size ${size} selected. `;
            const confirm = document.createElement('button');
            confirm.type = 'button'; confirm.className = 'confirm-choice'; confirm.textContent = 'Confirm this choice';
            confirm.addEventListener('click', async () => {
              confirm.disabled = true; status.textContent = 'Confirming your choice and finding a product image...';
              try {
                const result = await selectionRequest('confirm', item, size);
                const message = `Confirmed — you chose ${item.title} in size ${size}. I’ll remember this for future recommendations.`;
                bubble('agent', message, {confirmed_product: {...item, size, image_url: result.image_url || ''}});
                status.textContent = result.image_url ? 'Choice saved' : 'Choice saved — image search was unavailable';
                await speakWithAvatar(message);
              } catch (error) {
                bubble('agent', `I saved your selection, but could not finish the preview: ${error.message}`);
                status.textContent = 'Please try confirming again'; confirm.disabled = false;
              }
            });
            actions.append(summary, confirm);
          } catch (error) { bubble('agent', `I could not save that size: ${error.message}`); }
        });
        actions.appendChild(button);
      });
      box.appendChild(actions); chat.scrollTop = chat.scrollHeight;
    } catch (error) { bubble('agent', `I could not start the selection: ${error.message}`); }
  }

  async function submitMessage(message) {
    if (!message || !selectedFlow) return;
    bubble('user', message); input.value = ''; send.disabled = true;
    status.textContent = 'Thinking of the best answer for you...';
    try {
      const response = await fetch('/api/chat', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          session_id: sessionId, client_id: clientId,
          message, turn, top_k: 5, initial_intent: selectedFlow
        })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      bubble('agent', payload.message, payload); turn += 1; status.textContent = 'Ready when you are';
      await speakWithAvatar(payload.message);
    } catch (error) { bubble('agent', `Sorry, something went wrong: ${error.message}`); status.textContent = 'Please try again'; }
    finally { send.disabled = false; input.focus(); }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    await submitMessage(input.value.trim());
  });

  function waitForIce(pc) {
    if (pc.iceGatheringState === 'complete') return Promise.resolve();
    return new Promise(resolve => {
      const check = () => {
        if (pc.iceGatheringState === 'complete') {
          pc.removeEventListener('icegatheringstatechange', check); resolve();
        }
      };
      pc.addEventListener('icegatheringstatechange', check);
    });
  }

  function setEggMouth(level) {
    const open = Math.max(0, Math.min(1, level));
    smoothedMouthLevel += (open - smoothedMouthLevel) * .42;
    const height = 9 + smoothedMouthLevel * 28;
    const width = 36 + smoothedMouthLevel * 8;
    eggMouth.style.height = `${height}px`;
    eggMouth.style.width = `${width}px`;
    eggMouth.style.left = `${90 - width / 2}px`;
    eggMouth.style.borderRadius = smoothedMouthLevel > .14 ? '45%' : '4px 4px 26px 26px';
    avatarStage.classList.toggle('speaking', smoothedMouthLevel > .06);
  }

  function attachAudioLipSync(stream) {
    if (!window.AudioContext && !window.webkitAudioContext) return;
    audioContext ||= new (window.AudioContext || window.webkitAudioContext)();
    if (remoteAudioSource) remoteAudioSource.disconnect();
    remoteAudioSource = audioContext.createMediaStreamSource(stream);
    lipAnalyser = audioContext.createAnalyser();
    lipAnalyser.fftSize = 256;
    lipAnalyser.smoothingTimeConstant = .55;
    remoteAudioSource.connect(lipAnalyser);
    const samples = new Uint8Array(lipAnalyser.fftSize);
    cancelAnimationFrame(lipAnimationFrame);
    const animate = () => {
      lipAnalyser.getByteTimeDomainData(samples);
      let energy = 0;
      for (const sample of samples) {
        const normalized = (sample - 128) / 128;
        energy += normalized * normalized;
      }
      const rms = Math.sqrt(energy / samples.length);
      if (!fallbackMouthTimer) {
        setEggMouth(Math.max(0, Math.min(1, (rms - .012) * 11)));
      }
      lipAnimationFrame = requestAnimationFrame(animate);
    };
    animate();
  }

  function startFallbackMouth() {
    clearInterval(fallbackMouthTimer);
    let phase = 0;
    fallbackMouthTimer = setInterval(() => {
      phase += .9;
      setEggMouth(.2 + Math.abs(Math.sin(phase)) * .65);
    }, 90);
  }

  function stopFallbackMouth() {
    clearInterval(fallbackMouthTimer);
    fallbackMouthTimer = null;
    setEggMouth(0);
  }

  async function connectAvatar() {
    if (avatarPC) { avatarPC.close(); avatarPC = null; avatarSessionId = null; }
    const base = liveUrl.value.trim().replace(/\/$/, '');
    avatarConnect.disabled = true; avatarStatus.textContent = 'Waking up your assistant...';
    try {
      const pc = new RTCPeerConnection({sdpSemantics: 'unified-plan'});
      avatarPC = pc;
      pc.addTransceiver('video', {direction: 'recvonly'});
      pc.addTransceiver('audio', {direction: 'recvonly'});
      pc.addEventListener('track', event => {
        const stream = event.streams[0] || new MediaStream([event.track]);
        if (event.track.kind === 'video') avatarVideo.srcObject = stream;
        if (event.track.kind === 'audio') {
          avatarAudio.srcObject = stream;
          avatarAudio.muted = false;
          avatarAudio.volume = 1;
          attachAudioLipSync(stream);
        }
      });
      pc.addEventListener('connectionstatechange', () => {
        if (pc.connectionState === 'connected') {
          avatarDot.classList.add('online'); avatarStatus.textContent = 'NAmazon is here with you';
        } else if (['failed', 'disconnected', 'closed'].includes(pc.connectionState)) {
          avatarDot.classList.remove('online'); avatarStatus.textContent = 'Assistant is reconnecting';
        }
      });
      await pc.setLocalDescription(await pc.createOffer());
      await waitForIce(pc);
      const response = await fetch(`${base}/offer`, {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          sdp: pc.localDescription.sdp, type: pc.localDescription.type,
          avatar: 'namazon_ai_face',
          refaudio: 'en-US-JennyNeural'
        })
      });
      const answer = await response.json();
      if (!response.ok || !answer.sdp) throw new Error(answer.msg || `HTTP ${response.status}`);
      avatarSessionId = String(answer.sessionid);
      await pc.setRemoteDescription({type: 'answer', sdp: answer.sdp});
      avatarStatus.textContent = 'Almost ready...';
    } catch (error) {
      if (avatarPC) avatarPC.close(); avatarPC = null; avatarSessionId = null;
      avatarDot.classList.remove('online');
      avatarStatus.textContent = 'The animated assistant is unavailable';
      bubble('agent', 'My animated view is taking a short break, but you can still chat or use your voice.');
    } finally { avatarConnect.disabled = false; }
  }

  async function speakWithAvatar(text) {
    if (avatarSessionId) {
      const base = liveUrl.value.trim().replace(/\/$/, '');
      try {
        if (audioContext?.state === 'suspended') await audioContext.resume();
        await avatarAudio.play().catch(() => undefined);
        const response = await fetch(`${base}/human`, {
          method: 'POST', headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({sessionid: avatarSessionId, text, type: 'echo', interrupt: true})
        });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return;
      } catch (error) { avatarStatus.textContent = 'Voice playback is temporarily unavailable'; }
    }
    if ('speechSynthesis' in window) {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-US'; utterance.rate = 1.02;
      const englishVoices = speechSynthesis.getVoices().filter(voice => voice.lang.startsWith('en'));
      utterance.voice = englishVoices.find(voice => /Samantha|Jenny|Ava|Siri/i.test(voice.name)) || englishVoices[0] || null;
      utterance.onstart = startFallbackMouth;
      utterance.onboundary = () => setEggMouth(.75);
      utterance.onend = stopFallbackMouth;
      utterance.onerror = stopFallbackMouth;
      speechSynthesis.speak(utterance);
    } else { startFallbackMouth(); setTimeout(stopFallbackMouth, 2500); }
  }

  avatarConnect.addEventListener('click', connectAvatar);

  function microphoneError(error) {
    if (!window.isSecureContext) return 'Open this page on localhost or HTTPS to use the microphone.';
    if (error?.name === 'NotAllowedError') return 'Microphone permission is blocked. Allow it in your browser settings, then try again.';
    if (error?.name === 'NotFoundError') return 'No microphone was found. Connect one and try again.';
    if (error?.name === 'NotReadableError') return 'Your microphone is being used by another app.';
    return error?.message || 'The microphone could not be started.';
  }

  async function prepareMicrophone() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('This browser does not provide microphone access.');
    }
    if (!micPermissionReady) {
      status.textContent = 'Please allow microphone access...';
      const permissionStream = await navigator.mediaDevices.getUserMedia({
        audio: {echoCancellation: true, noiseSuppression: true, autoGainControl: true}
      });
      const activeDevice = permissionStream.getAudioTracks()[0]?.getSettings().deviceId || '';
      permissionStream.getTracks().forEach(track => track.stop());
      const devices = (await navigator.mediaDevices.enumerateDevices())
        .filter(device => device.kind === 'audioinput');
      micDevice.replaceChildren(...devices.map((device, index) => {
        const option = document.createElement('option');
        option.value = device.deviceId;
        option.textContent = device.label || `Microphone ${index + 1}`;
        option.selected = device.deviceId === activeDevice;
        return option;
      }));
      voiceControls.hidden = devices.length < 2;
      micPermissionReady = true;
    }
  }

  async function waitForSpeechWorker() {
    const deadline = Date.now() + 45000;
    while (Date.now() < deadline) {
      const response = await fetch(`/api/livekit/transcript?session_id=${encodeURIComponent(sessionId)}`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
      if (payload.error) throw new Error(payload.error);
      if (payload.status === 'ready') return;
      status.textContent = 'Preparing voice recognition...';
      await new Promise(resolve => setTimeout(resolve, 250));
    }
    throw new Error('Voice recognition took too long to start.');
  }

  async function connectLiveKitMic() {
    if (!window.LivekitClient) throw new Error('Voice service is not available');
    if (livekitRoom?.state === LivekitClient.ConnectionState.Connected) return livekitRoom;
    if (livekitConnecting) return livekitConnecting;
    livekitConnecting = (async () => {
      status.textContent = 'Preparing your microphone...';
      const response = await fetch('/api/livekit/token', {
        method: 'POST', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({session_id: sessionId})
      });
      const config = await response.json();
      if (!response.ok) throw new Error(config.error || `HTTP ${response.status}`);
      await waitForSpeechWorker();
      const room = new LivekitClient.Room({adaptiveStream: true, dynacast: true});
      room.on(LivekitClient.RoomEvent.Disconnected, () => {
        livekitRoom = null; livekitTrack = null; mic.classList.remove('listening');
      });
      await room.connect(config.url, config.token);
      livekitRoom = room;
      status.textContent = 'Microphone ready';
      return room;
    })();
    try { return await livekitConnecting; }
    finally { livekitConnecting = null; }
  }

  async function waitForLiveKitTranscript() {
    const deadline = Date.now() + 30000;
    while (Date.now() < deadline) {
      const response = await fetch(`/api/livekit/transcript?session_id=${encodeURIComponent(sessionId)}`);
      const payload = await response.json();
      if (payload.transcript) {
        input.value = payload.transcript;
        status.textContent = 'Got it — one moment...';
        await submitMessage(payload.transcript.trim());
        return;
      }
      if (payload.error) throw new Error(payload.error);
      status.textContent = payload.status === 'transcribing'
        ? 'Turning your voice into a message...'
        : 'Listening for your message...';
      await new Promise(resolve => setTimeout(resolve, 400));
    }
    throw new Error('Timed out waiting for the transcript');
  }

  async function startLiveKitRecording() {
    await prepareMicrophone();
    const room = await connectLiveKitMic();
    livekitTrack = await LivekitClient.createLocalAudioTrack({
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
      ...(micDevice.value ? {deviceId: micDevice.value} : {}),
    });
    await room.localParticipant.publishTrack(livekitTrack, {
      source: LivekitClient.Track.Source.Microphone,
    });
    status.textContent = 'Connecting your microphone...';
    await new Promise(resolve => setTimeout(resolve, 700));
    mic.classList.add('listening');
    mic.title = 'Stop recording and transcribe';
    status.textContent = 'Listening — tap the microphone again when you are done';
  }

  async function stopLiveKitRecording() {
    const track = livekitTrack;
    if (!track || !livekitRoom) return;
    livekitTrack = null;
    mic.classList.remove('listening');
    mic.title = 'Start a voice message';
    await livekitRoom.localParticipant.unpublishTrack(track);
    track.stop();
    status.textContent = 'Turning your voice into a message...';
    await waitForLiveKitTranscript();
  }

  mic.addEventListener('click', async () => {
    mic.disabled = true;
    try {
      if (livekitTrack) await stopLiveKitRecording();
      else await startLiveKitRecording();
    } catch (error) {
      mic.classList.remove('listening');
      status.textContent = 'Microphone unavailable';
      bubble('agent', microphoneError(error));
    } finally { mic.disabled = false; }
  });

  document.querySelectorAll('[data-flow]').forEach(button => {
    button.addEventListener('click', () => {
      selectedFlow = button.dataset.flow;
      flowPicker.hidden = true;
      input.disabled = false;
      send.disabled = false;
      mic.disabled = !window.LivekitClient;
      input.placeholder = selectedFlow === 'buying'
        ? 'Tell me what you want to buy, including requirements or budget...'
        : 'Tell me what category, style or idea you want to explore...';
      bubble('agent', selectedFlow === 'buying'
        ? 'Lovely — what are you hoping to find today?'
        : 'What are you in the mood to discover?');
      status.textContent = 'Ready when you are';
      input.focus();
      if (!avatarSessionId && !avatarPC) connectAvatar();
    });
  });

  document.querySelector('#reset').addEventListener('click', () => {
    if (livekitTrack && livekitRoom) {
      livekitRoom.localParticipant.unpublishTrack(livekitTrack); livekitTrack.stop(); livekitTrack = null;
    }
    if (livekitRoom) { livekitRoom.disconnect(); livekitRoom = null; }
    sessionId = crypto.randomUUID(); turn = 1; selectedFlow = null; chat.replaceChildren();
    bubble('agent', 'Fresh start! How can I help you today?');
    flowPicker.hidden = false; input.value = ''; input.disabled = true; send.disabled = true; mic.disabled = true;
    input.placeholder = 'Choose an option above to begin...'; status.textContent = 'Choose how you would like to start';
  });
</script>
</body>
</html>"""


class DemoServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], agent: Agent) -> None:
        # Initialize cleanup state before binding so server_close remains safe
        # even when the requested port cannot be opened.
        self.livekit_lock = RLock()
        self.livekit_workers: dict[str, subprocess.Popen[bytes]] = {}
        super().__init__(address, Handler)
        self.agent = agent
        self.project_root = Path(__file__).resolve().parent
        self.products = {str(row["parent_asin"]): row for row in agent.retriever.products}
        self.sessions: set[str] = set()
        self.session_clients: dict[str, str] = {}
        self.behavior_store = BehaviorStore(self.project_root / ".cache/user_behavior.json")
        self.product_images = ProductImageCache(self.project_root / ".cache/product_images")
        self.session_lock = RLock()
        self.livekit_secret = secrets.token_urlsafe(32)
        self.livekit_events: dict[str, list[dict[str, Any]]] = {}
        self.livekit_status: dict[str, str] = {}

    @staticmethod
    def livekit_room(session_id: str) -> str:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "", session_id)[:64]
        return f"shopping-{safe_id}"

    def ensure_livekit_worker(self, session_id: str) -> None:
        with self.livekit_lock:
            current = self.livekit_workers.get(session_id)
            if current is not None and current.poll() is None:
                return
            callback = f"http://127.0.0.1:{self.server_address[1]}/api/livekit/transcript"
            command = [
                sys.executable,
                str(self.project_root / "livekit_stt_worker.py"),
                "--room",
                self.livekit_room(session_id),
                "--session-id",
                session_id,
                "--callback",
                callback,
                "--callback-secret",
                self.livekit_secret,
            ]
            self.livekit_status[session_id] = "starting"
            self.livekit_workers[session_id] = subprocess.Popen(
                command,
                cwd=self.project_root,
            )

    def server_close(self) -> None:
        with self.livekit_lock:
            for process in self.livekit_workers.values():
                if process.poll() is None:
                    process.terminate()
        super().server_close()


class Handler(BaseHTTPRequestHandler):
    server: DemoServer

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} {format % args}", flush=True)

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, "application/json; charset=utf-8", json.dumps(payload, ensure_ascii=False).encode())

    def _read_json(self, max_size: int = 64_000) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > max_size:
            raise ValueError("Invalid request size")
        payload = json.loads(self.rfile.read(length))
        if not isinstance(payload, dict):
            raise ValueError("JSON object required")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif parsed.path == "/static/vendor/livekit-client.umd.min.js":
            sdk_path = self.server.project_root / "static/vendor/livekit-client.umd.min.js"
            self._send(200, "text/javascript; charset=utf-8", sdk_path.read_bytes())
        elif parsed.path == "/api/health":
            self._json(200, {
                "status": "ok",
                "products": len(self.server.products),
                "livekit_url": "ws://127.0.0.1:7880",
            })
        elif parsed.path == "/api/livekit/transcript":
            session_id = str(parse_qs(parsed.query).get("session_id", [""])[0]).strip()[:120]
            if not session_id:
                self._json(400, {"error": "session_id is required"})
                return
            with self.server.livekit_lock:
                events = self.server.livekit_events.setdefault(session_id, [])
                event = events.pop(0) if events else {}
                event.setdefault("status", self.server.livekit_status.get(session_id, "starting"))
            self._json(200, event)
        elif parsed.path == "/api/product/image":
            asin = str(parse_qs(parsed.query).get("asin", [""])[0]).strip()[:32]
            if not asin or asin not in self.server.products:
                self._json(404, {"error": "Product image not found"})
                return
            cached = self.server.product_images.cached(asin)
            if not cached:
                self._json(404, {"error": "Product image not found"})
                return
            image_path, mime_type = cached
            self._send(200, mime_type, image_path.read_bytes())
        else:
            self._json(404, {"error": "Not found"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            if self.path == "/api/livekit/token":
                payload = self._read_json()
                session_id = str(payload.get("session_id", "")).strip()[:120]
                if not session_id:
                    raise ValueError("session_id is required")
                room_name = self.server.livekit_room(session_id)
                self.server.ensure_livekit_worker(session_id)
                token = (
                    api.AccessToken("devkey", "secret")
                    .with_identity(f"web-{session_id[:48]}")
                    .with_name("Shopping user")
                    .with_grants(
                        api.VideoGrants(
                            room_join=True,
                            room=room_name,
                            can_publish=True,
                            can_subscribe=False,
                            can_publish_data=False,
                        )
                    )
                    .to_jwt()
                )
                self._json(200, {
                    "url": "ws://127.0.0.1:7880",
                    "token": token,
                    "room": room_name,
                })
                return

            if self.path == "/api/livekit/transcript":
                if not secrets.compare_digest(
                    self.headers.get("X-Worker-Secret", ""), self.server.livekit_secret
                ):
                    self._json(403, {"error": "Invalid worker secret"})
                    return
                payload = self._read_json()
                session_id = str(payload.get("session_id", "")).strip()[:120]
                if not session_id:
                    raise ValueError("session_id is required")
                event = {
                    key: payload[key]
                    for key in ("status", "transcript", "error", "duration")
                    if key in payload
                }
                with self.server.livekit_lock:
                    status = str(event.get("status", "ready"))
                    self.server.livekit_status[session_id] = status
                    if event.get("transcript") or event.get("error"):
                        self.server.livekit_events.setdefault(session_id, []).append(event)
                self._json(200, {"status": "ok"})
                return

            if self.path == "/api/selection":
                payload = self._read_json()
                session_id = str(payload.get("session_id", "")).strip()[:120]
                client_id = str(payload.get("client_id", "")).strip()[:120]
                asin = str(payload.get("parent_asin", "")).strip()[:32]
                action = str(payload.get("action", "")).strip().lower()
                size = str(payload.get("size", "")).strip()[:24]
                if session_id not in self.server.sessions:
                    raise ValueError("Unknown session")
                if not re.fullmatch(r"[A-Za-z0-9_-]{8,120}", client_id):
                    raise ValueError("Invalid client_id")
                if self.server.session_clients.get(session_id) != client_id:
                    raise ValueError("Session and client do not match")
                product = self.server.products.get(asin)
                if product is None:
                    raise ValueError("Unknown product")
                if action not in {"select", "size", "confirm"}:
                    raise ValueError("Invalid selection action")
                if action in {"size", "confirm"} and size not in {
                    "XS", "S", "M", "L", "XL", "One size"
                }:
                    raise ValueError("Invalid size")

                title = str(product.get("title", ""))
                if action == "select":
                    self.server.agent.record_behavior(
                        session_id,
                        f"The user selected {title} for closer consideration.",
                        selected_product=asin,
                    )
                elif action == "size":
                    self.server.agent.record_behavior(
                        session_id,
                        f"The user selected size {size} for {title}.",
                        selected_product=asin,
                        selected_size=size,
                        current_step="size_selected",
                    )
                else:
                    self.server.agent.record_behavior(
                        session_id,
                        f"The user confirmed {title} in size {size}. Treat this as a strong preference signal.",
                        selected_product=asin,
                        selected_size=size,
                        current_step="confirmed",
                    )
                    self.server.behavior_store.record(client_id, product, size)
                    found = self.server.product_images.find(product)
                    self._json(200, {
                        "status": "confirmed",
                        "image_url": f"/api/product/image?asin={asin}" if found else None,
                    })
                    return
                self._json(200, {"status": "ok"})
                return

            if self.path != "/api/chat":
                self._json(404, {"error": "Not found"})
                return

            payload = self._read_json()
            session_id = str(payload.get("session_id", "")).strip()[:120]
            client_id = str(payload.get("client_id", "")).strip()[:120]
            message = str(payload.get("message", "")).strip()[:4000]
            initial_intent = str(payload.get("initial_intent", "")).strip().lower()
            turn = max(1, min(int(payload.get("turn", 1)), 10))
            top_k = max(1, min(int(payload.get("top_k", 5)), 10))
            if not session_id or not message:
                raise ValueError("session_id and message are required")
            if not re.fullmatch(r"[A-Za-z0-9_-]{8,120}", client_id):
                raise ValueError("client_id is required")
            if initial_intent not in {"buying", "browsing"}:
                initial_intent = ""
            with self.server.session_lock:
                if session_id not in self.server.sessions:
                    behavior_summary = self.server.behavior_store.summary(client_id)
                    profile_summary = "Prefers comfort and durability."
                    if behavior_summary:
                        profile_summary += " " + behavior_summary
                    self.server.agent.reset(session_id, {
                        "summary": profile_summary,
                        "preference_tags": ["comfort", "durability"],
                        "rating_style": "usually positive",
                    })
                    self.server.sessions.add(session_id)
                    self.server.session_clients[session_id] = client_id
                    if initial_intent:
                        self.server.agent.memory.set_initial_intent(session_id, initial_intent)
                elif self.server.session_clients.get(session_id) != client_id:
                    raise ValueError("Session and client do not match")
            result = self.server.agent.respond(session_id, message, turn, top_k)
            for item in result.get("recommendations", []):
                product = self.server.products.get(str(item["parent_asin"]), {})
                item["title"] = str(product.get("title", ""))
                item["price"] = product.get("price")
            self._json(200, result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._json(400, {"error": str(exc)})
        except Exception as exc:  # Keep the local demo responsive and expose actionable errors.
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--catalog", default="data/catalog.jsonl")
    args = parser.parse_args()
    print("Loading the shopping catalog...", flush=True)
    server = DemoServer((args.host, args.port), Agent(args.catalog))
    print(f"Open http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
