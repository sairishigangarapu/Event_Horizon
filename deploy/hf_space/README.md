---
title: Crowd Disaster Information Reliability
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: Crowd-emergency info freshness and dual forecasts
---

# Crowd Disaster Information Reliability

Evaluator demo of the MKCrusaders26 project (Manipal Hackathon 2026). This Space is deployed automatically from the project's GitHub repository; edit the code there, not here.

- **Dashboard:** this page.
- **API documentation:** `/docs`.
- **Health and model status:** `/health`.

All data is synthetic, produced by the project's crowd simulator. Stream A is a simulated central forecast built only from observations. Stream B comes from the project's architecture pipeline. Only its decision-tree bank is fitted; the hypernetwork, GNN and GRU run with untrained weights, and the app labels them as experimental.

Free Spaces pause when idle, so the first visit after a pause can take a minute to start.
