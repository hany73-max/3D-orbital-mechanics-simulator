# 🌍 3D Orbital Mechanics Simulator — Python Project

---

## 🧠 Project Overview
This project presents an **interactive 3D orbital mechanics simulator** built using **Python**, **Streamlit**, **NumPy**, and **Plotly**.  
It visualizes satellite motion in an **Earth-centered inertial frame** using **Lagrange’s f and g functions**.  

The app provides real-time simulation of orbital motion, allowing users to modify orbital parameters through an intuitive web interface and instantly observe the orbit’s geometry and dynamics.

---

## 🚀 Features
- Real-time 3D orbit visualization using Plotly  
- Adjustable orbital parameters:
  - Semi-major axis *(a)*
  - Eccentricity *(e)*
  - Inclination *(i)*
  - Right Ascension of Ascending Node *(RAAN)*
  - Argument of Periapsis *(ω)*
- Animated satellite motion along its orbit  
- Calculation of orbital period and velocity vectors  
- Built with modern interactive tools: **Streamlit** + **Plotly**

---

## ⚙️ How to Run the App

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/orbital-mechanics-simulator.git
   cd orbital-mechanics-simulator
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit numpy plotly
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

4. The application will open automatically in your default web browser.  
   You can now adjust the parameters and view the orbit in 3D.

---

## 🧩 Project Structure
```
Orbital_Mechanics_Simulator/
│
├── app.py                     # Main Streamlit app
└── README.md                  # Project documentation
```

---

## 📚 Theoretical Background
The simulation is based on **Lagrange’s equations**, which describe orbital motion as functions of time and true anomaly.  
These relationships allow propagation of position and velocity vectors from initial orbital parameters.

---