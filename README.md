# Tic-Tac-Toe Reinforcement Learning Projekt

Ein Java-Projekt zur Erforschung verschiedener Reinforcement-Learning-Algorithmen am Beispiel von Tic-Tac-Toe. Das Projekt nutzt die [Neuroph](http://neuroph.sourceforge.net/) Bibliothek für neuronale Netzwerke.

## Inhaltsverzeichnis

- [Übersicht](#übersicht)
- [Spieler-Implementierungen](#spieler-implementierungen)
  - [QLearningNeurophSpieler](#qlearningneurophspieler)
  - [PolicyGradientNeurophSpieler](#policygradientneurophspieler)
  - [CopycatSpieler](#copycatspieler)
  - [Tabellenbasierte Spieler](#tabellenbasierte-spieler)
- [Neuronale Netzwerk-Architektur](#neuronale-netzwerk-architektur)
- [Neuronales Netzwerk vs. Tabellenbasiertes Q-Learning](#neuronales-netzwerk-vs-tabellenbasiertes-q-learning)
- [Trainings-Loop im Detail](#trainings-loop-im-detail)
- [Installation & Ausführung](#installation--ausführung)
- [Projektstruktur](#projektstruktur)

---

## Übersicht

Dieses Projekt implementiert mehrere KI-Agenten, die Tic-Tac-Toe durch Selbstspiel und Spiel gegen einen Zufallsspieler erlernen. Die Hauptziele sind:

1. **Vergleich verschiedener RL-Algorithmen** (Q-Learning, Policy Gradient, episodisches Lernen)
2. **Untersuchung von Netzwerk-Architekturen** (Größe, Aktivierungsfunktionen, Encoding)
3. **Praktische Demonstration** von Konzepten wie Experience Replay, ε-greedy Exploration, Discount-Faktoren

---

## Spieler-Implementierungen

### QLearningNeurophSpieler

Der robusteste und empfohlene Spieler. Implementiert **Deep Q-Learning (DQN)** mit modernen Techniken:

**Algorithmus:**
```
Q(s,a) ← r + γ × max_a' Q(s', a')
```

**Kernfeatures:**
- ✅ **Experience Replay Buffer** (10.000 Erfahrungen) - Zufälliges Sampling für stabiles Training
- ✅ **Batch Training** (32 Samples) - Effizientes Lernen
- ✅ **Exponentieller ε-Decay** - Von 1.0 auf 0.05 über 100.000 Runden
- ✅ **Differenziertes Reward-System** (+1 Sieg, -1 Niederlage, +0.3 Unentschieden)
- ✅ **TANH-Aktivierung** - Ermöglicht negative Q-Werte

**Architektur:** `9 → 36 → 36 → 9`

**Hyperparameter:**
| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| GAMMA | 0.95 | Discount-Faktor |
| ALPHA | 0.01 | Lernrate |
| BATCH_SIZE | 32 | Samples pro Update |
| REPLAY_BUFFER | 10.000 | Max. gespeicherte Erfahrungen |

---

### PolicyGradientNeurophSpieler

Implementiert den **REINFORCE-Algorithmus** (Monte-Carlo Policy Gradient):

**Algorithmus:**
```
∇J(θ) = E[∇log π(a|s) × G_t]
```

**Kernfeatures:**
- ✅ **Stochastische Policy** - Natürliche Exploration durch Wahrscheinlichkeitsverteilung
- ✅ **Entropy-Regularisierung** - Verhindert vorzeitige Konvergenz
- ✅ **Monte-Carlo Returns** - Lernt von kompletten Episoden
- ❌ **Kein Experience Replay** - On-policy Algorithmus

**Architektur:** `9 → 64 → 64 → 9` (größer für komplexere Policy-Repräsentation)

**Hyperparameter:**
| Parameter | Wert | Bedeutung |
|-----------|------|-----------|
| GAMMA | 0.99 | Discount (höher für langfristige Credits) |
| LR | 0.01 | Lernrate |
| BETA | 0.5 | Skalierung der Policy-Updates |
| ENTROPY_EPS | 0.02 | Stärke der Regularisierung |

**Wann Policy Gradient?**
- Kontinuierliche Aktionsräume (hier nicht relevant)
- Wenn stochastische Policies gewünscht sind
- Bei Problemen mit overestimation in Q-Learning

---

### CopycatSpieler

Ein **vereinfachter Q-Learning Spieler** für Experimente:

**Besonderheiten:**
- **One-Hot Encoding** (27 Inputs) - Bessere Feature-Trennung
- **Episodisches Batch-Update** - Kein Experience Replay
- **Linearer ε-Decay**

**Architektur:** `27 → 9 → 9 → 9 → 9`

**⚠️ Bekannte Probleme:**
- Multipliziert Lernrate in das Target (nicht standard)
- Nur binärer Reward (0/1)
- Kein Replay Buffer → instabiles Lernen

**Empfehlung:** Für schnelle Experimente, aber für ernsthaftes Training `QLearningNeurophSpieler` verwenden.

---

### Tabellenbasierte Spieler

**QLearningSpielerRandom** und **QLearningSpielerHeuristik** nutzen klassisches tabellarisches Q-Learning:

```java
HashMap<String, double[]> qTable = new HashMap<>();
```

Speichern explizit Q-Werte für jeden besuchten Zustand.

---

## Neuronale Netzwerk-Architektur

### Warum diese Layer-Größen?

**QLearningNeurophSpieler: 9-36-36-9**

```
Eingabe (9)     Hidden 1 (36)    Hidden 2 (36)    Ausgabe (9)
   ●               ●               ●               ●
   ●               ●               ●               ●
   ●       →       ●       →       ●       →       ●
   ●               ●               ●               ●
   ...             ...             ...             ...
```

- **9 Inputs:** Ein Wert pro Spielfeld-Position (-1/0/+1)
- **36 Hidden:** 4× Eingabegröße - erfasst Kombinationen von Feldern (Linien, Ecken, Zentrum)
- **9 Outputs:** Q-Wert für jeden möglichen Zug

**PolicyGradientNeurophSpieler: 9-64-64-9**

- Größere Hidden-Layer (64) für komplexere Policy-Repräsentation
- Policy-Netzwerke profitieren oft von mehr Kapazität

### Aktivierungsfunktionen

| Funktion | Bereich | Verwendung | Warum |
|----------|---------|------------|-------|
| **TANH** | [-1, +1] | Q-Learning | Negative Q-Werte für Verluste möglich |
| **SIGMOID** | [0, 1] | Policy Gradient | Interpretierbar als Wahrscheinlichkeiten |

### Board-Encoding Strategien

**Signed Encoding (QLearningNeurophSpieler):**
```
Position 0-8: +1.0 (Agent), -1.0 (Gegner), 0.0 (Leer)
Kompakt: 9 Werte
```

**One-Hot Encoding (CopycatSpieler):**
```
Position 0-8:   1.0 wenn Leer
Position 9-17:  1.0 wenn Agent
Position 18-26: 1.0 wenn Gegner
Expandiert: 27 Werte
```

**Neutral Encoding (PolicyGradientNeurophSpieler):**
```
Position 0-8: 1.0 (Agent), 0.0 (Gegner), 0.5 (Leer)
Kompakt: 9 Werte, aber verschobener Nullpunkt
```

---

## Neuronales Netzwerk vs. Tabellenbasiertes Q-Learning

### Vergleich

| Aspekt | Tabellen-Q-Learning | Neural Network Q-Learning |
|--------|---------------------|---------------------------|
| **Speicher** | O(|S| × |A|) - explizit | O(Gewichte) - kompakt |
| **Generalisierung** | Keine | Ja, ähnliche Zustände |
| **Tic-Tac-Toe Zustände** | ~5.478 erreichbar | ~3.000 Gewichte |
| **Trainingszeit** | Schnell | Langsamer |
| **Stabilität** | Garantiert konvergent | Kann instabil sein |
| **Skalierbarkeit** | Begrenzt | Beliebig große Zustandsräume |

### Wann welcher Ansatz?

**Tabellenbasiert (QLearningSpielerRandom):**
- ✅ Kleiner, diskreter Zustandsraum
- ✅ Garantierte Konvergenz gewünscht
- ✅ Interpretierbare Q-Werte
- ❌ Keine Generalisierung auf neue Zustände

**Neuronales Netzwerk (QLearningNeurophSpieler):**
- ✅ Großer/kontinuierlicher Zustandsraum
- ✅ Generalisierung auf ähnliche Situationen
- ✅ Kompakte Repräsentation
- ❌ Hyperparameter-Tuning erforderlich
- ❌ Kann "catastrophic forgetting" erleiden

### Für Tic-Tac-Toe

Beide Ansätze funktionieren. Neural Networks sind hier **Overkill**, aber lehrreich:
- Tic-Tac-Toe hat nur ~5.478 mögliche Zustände
- Eine Q-Tabelle mit 5.478 × 9 = ~50.000 Einträgen ist trivial
- NN übt aber Konzepte, die für komplexere Spiele (Schach, Go) essentiell sind

---

## Trainings-Loop im Detail

### QLearningNeurophSpieler Training

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│  1. Initialisiere Netzwerk & Replay Buffer                  │
│                                                             │
│  for runde in 0..∞ until abbruch:                          │
│    │                                                        │
│    ├─ 2. Simuliere Spiel gegen Zufallsspieler               │
│    │     │                                                  │
│    │     ├─ Agent wählt Aktion (ε-greedy)                  │
│    │     ├─ Speichere (s, a, r, s', terminal) im Buffer    │
│    │     └─ Wiederhole bis Spielende                       │
│    │                                                        │
│    ├─ 3. Alle 4 Spiele: trainFromReplay()                  │
│    │     │                                                  │
│    │     ├─ Sample 32 zufällige Erfahrungen                │
│    │     ├─ Berechne Q-Targets                             │
│    │     │   if terminal: target = reward                  │
│    │     │   else: target = r + γ × max Q(s',a')           │
│    │     └─ Trainiere Netzwerk mit Batch                   │
│    │                                                        │
│    └─ 4. ε *= decay_rate (exponentiell abnehmend)          │
│                                                             │
│  5. Speichere trainiertes Netzwerk                         │
└─────────────────────────────────────────────────────────────┘
```

### PolicyGradientNeurophSpieler Training

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                            │
├─────────────────────────────────────────────────────────────┤
│  for episode in 0..∞ until abbruch:                        │
│    │                                                        │
│    ├─ 1. Spiele komplette Episode, sammle Trajektorie      │
│    │     trajectory = [(s₀,a₀,r₀), (s₁,a₁,r₁), ...]       │
│    │                                                        │
│    ├─ 2. Berechne Returns rückwärts                        │
│    │     G_t = r_t + γ×G_{t+1}                             │
│    │                                                        │
│    └─ 3. trainOnTrajectory()                               │
│          │                                                  │
│          ├─ Für jeden Step:                                │
│          │   π'(a|s) = π(a|s) + β × G_t  (erhöhe wenn G>0) │
│          │   Normalisiere zu Wahrscheinlichkeiten          │
│          │   Mische mit Uniform (Entropy-Regularisierung)  │
│          │                                                  │
│          └─ Trainiere Netzwerk mit modifizierten Targets   │
└─────────────────────────────────────────────────────────────┘
```

### Reward-Struktur

| Spieler | Sieg | Niederlage | Unentschieden | Zwischen |
|---------|------|------------|---------------|----------|
| QLearningNeurophSpieler | +1.0 | -1.0 | +0.3 | 0.0 |
| PolicyGradientNeurophSpieler | +1.0 | -1.0 | 0.0 | 0.0 |
| CopycatSpieler | +1.0 | 0.0 | 0.0 | 0.0 |

**Warum +0.3 für Unentschieden?**
- Unentschieden ist bei optimalem Spiel das beste erreichbare Ergebnis
- Leicht positiver Reward ermutigt "nicht verlieren"
- Verhindert, dass Agent riskante Züge macht um zu gewinnen

---

## Installation & Ausführung

### Voraussetzungen

- Java 17+
- Maven 3.6+

### Build

```bash
cd TicTacToeFlender
mvn clean compile
```

### Training starten

```bash
# Q-Learning Spieler trainieren
mvn exec:java -Dexec.mainClass="de.whs.wi.tictactoe.QLearningTrainingRunner"

# Policy Gradient Spieler trainieren
mvn exec:java -Dexec.mainClass="de.whs.wi.tictactoe.NeurophLearningRunner"
```

### Wettkampf starten

```bash
mvn exec:java -Dexec.mainClass="de.whs.wi.Main"
```

---

## Projektstruktur

```
TicTacToeFlender/
├── pom.xml                          # Maven Konfiguration
├── README.md                        # Diese Datei
├── wissenNeurophZufall.bin          # Gespeichertes Q-Learning Netzwerk
├── wissenHeuristik.bin              # Gespeichertes Heuristik-Wissen
│
└── src/main/java/de/whs/wi/
    ├── Main.java                    # Haupteinstiegspunkt
    │
    └── tictactoe/
        ├── QLearningTrainingRunner.java        # Training für Q-Learning
        ├── NeurophLearningRunner.java          # Training für Neuroph-Spieler
        ├── QLearningHeuristicTrainingRunner.java
        │
        └── spieler/Flender/
            ├── QLearningNeurophSpieler.java    # ⭐ Bester NN-Spieler
            ├── PolicyGradientNeurophSpieler.java
            ├── CopycatSpieler.java
            ├── QLearningSpielerRandom.java     # Tabellenbasiert
            ├── QLearningSpielerHeuristik.java  # Tabellenbasiert + Heuristik
            ├── HeuristikSpieler.java           # Regelbasiert
            └── SpielHistoryEntry.java          # Hilfsklasse
```

---

## Weiterführende Ressourcen

- [Deep Q-Network (DQN) Paper](https://www.nature.com/articles/nature14236) - Mnih et al., 2015
- [Policy Gradient Methods](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) - OpenAI Spinning Up
- [Neuroph Documentation](http://neuroph.sourceforge.net/documentation.html)

---

## Lizenz

Dieses Projekt ist für Bildungszwecke erstellt.

## Autor

Flender

