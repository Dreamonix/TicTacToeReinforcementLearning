// Java
    // Fixes: strict valid masking, safe fallback in berechneZug, robust greedy selection.

    package de.whs.wi.tictactoe.spieler.Flender;

    import org.neuroph.core.NeuralNetwork;
    import org.neuroph.nnet.MultiLayerPerceptron;
    import org.neuroph.util.TransferFunctionType;
    import tictactoe.*;
    import tictactoe.spieler.IAbbruchbedingung;
    import tictactoe.spieler.ILernenderSpieler;

    import java.io.*;
    import java.util.*;

/**
 * Ein Policy-Gradient-basierter Tic-Tac-Toe-Spieler (REINFORCE-Algorithmus).
 *
 * <h2>Algorithmus-Übersicht</h2>
 * <p>Dieser Spieler implementiert den REINFORCE Policy-Gradient Algorithmus:</p>
 * <ul>
 *   <li><b>Stochastische Policy:</b> Netzwerk gibt Wahrscheinlichkeiten für Aktionen aus</li>
 *   <li><b>Monte-Carlo Returns:</b> Lernt von kompletten Episoden (kein Bootstrapping)</li>
 *   <li><b>Entropy Regularisierung:</b> Verhindert vorzeitige Konvergenz zu deterministischer Policy</li>
 * </ul>
 *
 * <h2>Netzwerk-Architektur</h2>
 * <p>9 → 64 → 64 → 9 mit SIGMOID-Aktivierung:</p>
 * <ul>
 *   <li><b>9 Inputs:</b> Spielfeld-Kodierung (0.5 = Leer, 1.0 = Agent, 0.0 = Gegner)</li>
 *   <li><b>64-64 Hidden:</b> Größere Kapazität für komplexe Policy-Muster</li>
 *   <li><b>9 Outputs:</b> Unnormalisierte Wahrscheinlichkeiten (werden normalisiert)</li>
 *   <li><b>SIGMOID:</b> Ausgabe [0,1] für Wahrscheinlichkeitsinterpretation</li>
 * </ul>
 *
 * <h2>Hyperparameter</h2>
 * <ul>
 *   <li>GAMMA = 0.99: Discount-Faktor (höher als Q-Learning für langfristige Credits)</li>
 *   <li>LR = 0.01: Lernrate für Backpropagation</li>
 *   <li>BETA = 0.5: Skalierungsfaktor für Policy-Gradient Updates</li>
 *   <li>ENTROPY_EPS = 0.02: Stärke der Entropy-Regularisierung</li>
 * </ul>
 *
 * <h2>Vergleich zu Q-Learning</h2>
 * <table border="1">
 *   <tr><th>Aspekt</th><th>Policy Gradient</th><th>Q-Learning</th></tr>
 *   <tr><td>Lernt</td><td>Policy π(a|s) direkt</td><td>Q-Funktion Q(s,a)</td></tr>
 *   <tr><td>Exploration</td><td>Stochastisch (natürlich)</td><td>ε-greedy (künstlich)</td></tr>
 *   <tr><td>Varianz</td><td>Hoch (Monte-Carlo)</td><td>Niedriger (TD)</td></tr>
 *   <tr><td>Sample-Effizienz</td><td>Geringer (on-policy)</td><td>Höher (off-policy, Replay)</td></tr>
 * </table>
 *
 * @author Flender
 * @see ILernenderSpieler
 * @see QLearningNeurophSpieler
 */
public class PolicyGradientNeurophSpieler implements ILernenderSpieler {

    /** Discount-Faktor für zukünftige Rewards (0.99 = sehr langfristig) */
    private static final double GAMMA = 0.99;

    /** Lernrate für Backpropagation */
    private static final double LR = 0.01;

    /** Skalierungsfaktor für Policy-Updates (höher = aggressivere Updates) */
    private static final double BETA = 0.5;

    /** Entropy-Regularisierung (mischt Uniform-Verteilung ein für Exploration) */
    private static final double ENTROPY_EPS = 0.02;

    /** Zufallsgenerator für stochastische Aktionswahl */
    private final Random rnd = new Random();

    /** Das neuronale Netzwerk für die Policy */
    private NeuralNetwork<?> policyNet;

    /** Name des Spielers */
    private String name;

    /** Farbe des Agenten im aktuellen Spiel */
    private Farbe agentFarbe;

    /** Internes Spielfeld zur Zugberechnung */
    private Spielfeld internesSpielfeld;

    /**
     * Erstellt einen neuen Policy-Gradient Spieler mit initialisiertem Netzwerk.
     *
     * @param name Der Name des Spielers
     */
    public PolicyGradientNeurophSpieler(String name) {
        this.name = name;
        initNetwork();
    }

    /**
     * Initialisiert das Policy-Netzwerk mit Architektur 9-64-64-9.
     *
     * <p>Konfiguration:</p>
     * <ul>
     *   <li>SIGMOID-Aktivierung für Wahrscheinlichkeits-Outputs</li>
     *   <li>Größere Hidden-Layer (64) als Q-Learning für Policy-Komplexität</li>
     *   <li>BackPropagation mit LR, Single-Iteration pro Batch</li>
     * </ul>
     */
    private void initNetwork() {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                TransferFunctionType.SIGMOID,
                9, 64, 64, 9
        );
        org.neuroph.nnet.learning.BackPropagation bp =
                (org.neuroph.nnet.learning.BackPropagation) mlp.getLearningRule();
        bp.setLearningRate(LR);
        bp.setMaxIterations(1);
        bp.setBatchMode(false);
        this.policyNet = mlp;
    }

    /**
     * Kodiert das Spielfeld als Eingabevektor für das Netzwerk.
     *
     * <p>Kodierungsschema (anders als Q-Learning):</p>
     * <ul>
     *   <li>0.5: Leeres Feld (neutral)</li>
     *   <li>1.0: Eigene Farbe (Agent)</li>
     *   <li>0.0: Gegnerische Farbe</li>
     * </ul>
     *
     * @param board Das zu kodierende Spielfeld
     * @param agent Die Farbe des Agenten
     * @return double[9] Kodierter Zustandsvektor
     */
    private double[] encode(Spielfeld board, Farbe agent) {
        double[] x = new double[9];
        int k = 0;
        for (int z = 0; z < 3; z++) {
            for (int s = 0; s < 3; s++) {
                Farbe f = board.getFarbe(z, s);
                if (f == Farbe.Leer) x[k] = 0.5;
                else if (f == agent) x[k] = 1.0;
                else x[k] = 0.0;
                k++;
            }
        }
        return x;
    }

    /**
     * Erstellt eine Maske der gültigen Aktionen.
     *
     * @param b Das aktuelle Spielfeld
     * @return boolean[9] True für jedes leere (spielbare) Feld
     */
    private boolean[] validMask(Spielfeld b) {
        boolean[] m = new boolean[9];
        for (int a = 0; a < 9; a++) m[a] = (b.getFarbe(a / 3, a % 3) == Farbe.Leer);
        return m;
    }

    /**
     * Berechnet die rohen Netzwerk-Ausgaben ohne Normalisierung.
     *
     * @param state Kodierter Spielzustand
     * @return double[9] Rohe Ausgabewerte [0,1] durch SIGMOID
     */
    private double[] rawOutput(double[] state) {
        policyNet.setInput(state);
        policyNet.calculate();
        double[] out = new double[9];
        int i = 0;
        for (Double d : policyNet.getOutput()) out[i++] = d;
        return out;
    }

    /**
     * Berechnet die normalisierte Policy-Verteilung für gültige Aktionen.
     *
     * <p>Algorithmus:</p>
     * <ol>
     *   <li>Hole rohe Netzwerk-Ausgabe</li>
     *   <li>Setze ungültige Aktionen auf 0</li>
     *   <li>Normalisiere zu Wahrscheinlichkeitsverteilung (Summe = 1)</li>
     *   <li>Falls alle Werte ~0: Uniform über gültige Aktionen</li>
     * </ol>
     *
     * @param state Kodierter Spielzustand
     * @param valid Maske der gültigen Aktionen
     * @return double[9] Normalisierte Wahrscheinlichkeiten
     */
    private double[] policyFor(double[] state, boolean[] valid) {
        double[] y = rawOutput(state);
        double sum = 0.0;
        for (int i = 0; i < 9; i++) {
            if (!valid[i]) y[i] = 0.0;
            sum += y[i];
        }
        if (sum <= 1e-12) {
            int cnt = 0;
            for (boolean v : valid) if (v) cnt++;
            double p = cnt > 0 ? 1.0 / cnt : 0.0;
            Arrays.fill(y, 0.0);
            for (int i = 0; i < 9; i++) if (valid[i]) y[i] = p;
            return y;
        }
        for (int i = 0; i < 9; i++) y[i] /= sum;
        return y;
    }

    /**
     * Sampelt eine Aktion gemäß der Policy-Verteilung.
     *
     * <p>Verwendet inverse CDF sampling:</p>
     * <ol>
     *   <li>Ziehe Zufallszahl r ∈ [0,1)</li>
     *   <li>Summiere Wahrscheinlichkeiten bis Summe > r</li>
     *   <li>Wähle entsprechende Aktion</li>
     * </ol>
     *
     * @param probs Wahrscheinlichkeitsverteilung
     * @param valid Maske der gültigen Aktionen
     * @return Gesampelte Aktion (0-8)
     */
    private int sampleAction(double[] probs, boolean[] valid) {
        // Ensure sampling only over valid actions
        double r = rnd.nextDouble(), cum = 0.0;
        for (int i = 0; i < probs.length; i++) {
            if (!valid[i]) continue;
            cum += probs[i];
            if (r <= cum) return i;
        }
        // Fallback to last valid with non-zero prob or first valid
        for (int i = probs.length - 1; i >= 0; i--) if (valid[i] && probs[i] > 0) return i;
        for (int i = 0; i < probs.length; i++) if (valid[i]) return i;
        return 0;
    }

    /**
     * Wählt die Aktion mit höchster Wahrscheinlichkeit (für Wettkampf).
     *
     * @param probs Wahrscheinlichkeitsverteilung
     * @param valid Maske der gültigen Aktionen
     * @return Beste gültige Aktion
     */
    private int greedyAction(double[] probs, boolean[] valid) {
        int best = -1;
        double v = -1.0;
        for (int i = 0; i < probs.length; i++) {
            if (!valid[i]) continue;
            if (probs[i] > v) { v = probs[i]; best = i; }
        }
        if (best != -1) return best;
        for (int i = 0; i < probs.length; i++) if (valid[i]) return i;
        return 0;
    }

    /**
     * Prüft ob das Spielfeld vollständig belegt ist.
     *
     * @param s Das zu prüfende Spielfeld
     * @return true wenn alle Felder belegt
     */
    private boolean voll(Spielfeld s) {
        for (int z = 0; z < 3; z++)
            for (int sp = 0; sp < 3; sp++)
                if (s.getFarbe(z, sp) == Farbe.Leer) return false;
        return true;
    }

    /**
     * Simuliert eine komplette Episode gegen einen Zufallsspieler.
     *
     * <p>Sammelt Trajektorie mit:</p>
     * <ul>
     *   <li>Zustand vor jedem Zug</li>
     *   <li>Gewählte Aktion</li>
     *   <li>Erhaltener Reward (+1 Sieg, -1 Niederlage, 0 sonst)</li>
     *   <li>Maske gültiger Aktionen</li>
     * </ul>
     *
     * @param agentStart Startfarbe des Agenten
     * @return Liste von Step-Objekten (Trajektorie)
     */
    private List<Step> episodeVsRandom(Farbe agentStart) {
        Spielfeld board = new Spielfeld();
        tictactoe.spieler.beispiel.Zufallsspieler rndOpp =
                new tictactoe.spieler.beispiel.Zufallsspieler("Zufall");
        Farbe agentCol = agentStart;
        Farbe oppCol = agentCol.opposite();
        Farbe amZug = Farbe.Kreuz;
        rndOpp.neuesSpiel(oppCol, 0);

        List<Step> traj = new ArrayList<>();
        Zug letzterZug = null;

        while (true) {
            if (amZug == agentCol) {
                double[] state = encode(board, agentCol);
                boolean[] mask = validMask(board);
                double[] pi = policyFor(state, mask);
                int a = sampleAction(pi, mask);
                board.setFarbe(a / 3, a % 3, agentCol);

                Spielstand win = board.pruefeGewinn(agentCol);
                if (win == Spielstand.GEWONNEN) {
                    traj.add(new Step(state, a, 1.0, mask));
                    break;
                }
                if (voll(board)) {
                    traj.add(new Step(state, a, 0.0, mask));
                    break;
                }
                traj.add(new Step(state, a, 0.0, mask));
                letzterZug = new Zug(a / 3, a % 3);

            } else {
                rndOpp.setFarbe(oppCol);
                Zug zug = rndOpp.berechneZug(letzterZug, 0, 0);
                board.setFarbe(zug.getZeile(), zug.getSpalte(), oppCol);

                Spielstand oppWin = board.pruefeGewinn(oppCol);
                if (oppWin == Spielstand.GEWONNEN) {
                    if (!traj.isEmpty()) {
                        traj.get(traj.size() - 1).reward = -1.0;
                    }
                    break;
                }
                if (voll(board)) break;
                letzterZug = zug;
            }
            amZug = amZug.opposite();
        }
        return traj;
    }

    /**
     * Trainiert das Netzwerk auf einer Trajektorie (REINFORCE-Update).
     *
     * <p>Algorithmus:</p>
     * <ol>
     *   <li>Berechne diskontierte Returns G_t = Σ γ^k * r_{t+k} rückwärts</li>
     *   <li>Für jeden Step: Erhöhe π(a|s) proportional zu G (wenn positiv)</li>
     *   <li>Normalisiere und wende Entropy-Regularisierung an</li>
     *   <li>Trainiere Netzwerk mit modifiziertem Target</li>
     * </ol>
     *
     * <p>Entropy-Regularisierung verhindert Kollaps zu deterministischer Policy
     * durch Beimischung einer Uniform-Verteilung.</p>
     *
     * @param traj Die zu lernende Trajektorie
     */
    private void trainOnTrajectory(List<Step> traj) {
        if (traj.isEmpty()) return;

        double G = 0.0;
        double[] returns = new double[traj.size()];
        for (int i = traj.size() - 1; i >= 0; i--) {
            G = traj.get(i).reward + GAMMA * G;
            returns[i] = G;
        }

        org.neuroph.core.data.DataSet ds = new org.neuroph.core.data.DataSet(9, 9);

        for (int i = 0; i < traj.size(); i++) {
            Step st = traj.get(i);
            double[] p = policyFor(st.state, st.validMask);
            double[] y = Arrays.copyOf(p, 9);

            y[st.action] = Math.max(0.0, y[st.action] + BETA * returns[i]);

            double sum = 0.0;
            for (int k = 0; k < 9; k++) {
                if (!st.validMask[k]) y[k] = 0.0;
                sum += y[k];
            }
            if (sum <= 1e-12) {
                int cnt = 0; for (boolean v : st.validMask) if (v) cnt++;
                double u = cnt > 0 ? 1.0 / cnt : 0.0;
                Arrays.fill(y, 0.0);
                for (int k = 0; k < 9; k++) if (st.validMask[k]) y[k] = u;
            } else {
                for (int k = 0; k < 9; k++) y[k] /= sum;
            }

            int cnt = 0; for (boolean v : st.validMask) if (v) cnt++;
            double u = cnt > 0 ? 1.0 / cnt : 0.0;
            for (int k = 0; k < 9; k++) {
                double uni = st.validMask[k] ? u : 0.0;
                y[k] = (1.0 - ENTROPY_EPS) * y[k] + ENTROPY_EPS * uni;
                y[k] = Math.max(1e-4, Math.min(1.0 - 1e-4, y[k]));
            }

            ds.add(st.state, y);
        }

        policyNet.learn(ds);
    }

    /**
     * Führt den Trainingsprozess durch bis die Abbruchbedingung erfüllt ist.
     *
     * <p>Training Loop (einfacher als Q-Learning):</p>
     * <ol>
     *   <li>Spiele Episode gegen Zufallsspieler</li>
     *   <li>Trainiere sofort auf dieser Episode (on-policy)</li>
     *   <li>Wiederhole mit alternierender Startfarbe</li>
     * </ol>
     *
     * <p>Kein Experience Replay nötig/möglich (on-policy Algorithmus).</p>
     *
     * @param abbruch Abbruchbedingung
     * @return true wenn Training erfolgreich
     */
    @Override
    public boolean trainieren(IAbbruchbedingung abbruch) {
        long ep = 0;
        long start = System.currentTimeMillis();
        do {
            Farbe starter = (ep % 2 == 0) ? Farbe.Kreuz : Farbe.Kreis;
            List<Step> traj = episodeVsRandom(starter);
            trainOnTrajectory(traj);

            ep++;
            if (ep % 10000 == 0) {
                long ms = System.currentTimeMillis() - start;
                double epsPerSec = ep / Math.max(1.0, (ms / 1000.0));
                System.out.printf("Episode %d | %.1f eps/sec%n", ep, epsPerSec);
            }
        } while (!abbruch.abbruch());
        return true;
    }

    /**
     * Initialisiert den Spieler für ein neues Spiel.
     *
     * @param farbe Die zugewiesene Farbe
     * @param bedenkzeitInSekunden Verfügbare Bedenkzeit (nicht verwendet)
     */
    @Override
    public void neuesSpiel(Farbe farbe, int bedenkzeitInSekunden) {
        this.agentFarbe = farbe;
        this.internesSpielfeld = new Spielfeld(); // reset every game
    }

    /**
     * Berechnet den nächsten Zug basierend auf der gelernten Policy.
     *
     * <p>Im Wettkampf wird greedy die wahrscheinlichste Aktion gewählt.
     * Enthält Fallback-Logik falls greedyAction ungültigen Zug liefert.</p>
     *
     * @param vorherigerZug Der letzte Zug des Gegners
     * @param zeitKreis Verbleibende Zeit für Kreis
     * @param zeitKreuz Verbleibende Zeit für Kreuz
     * @return Der berechnete Zug
     * @throws IllegalerZugException Bei keinen gültigen Zügen
     */
    @Override
    public Zug berechneZug(Zug vorherigerZug, long zeitKreis, long zeitKreuz) throws IllegalerZugException {
        if (vorherigerZug != null) {
            Farbe gegner = agentFarbe.opposite();
            internesSpielfeld.setFarbe(vorherigerZug.getZeile(), vorherigerZug.getSpalte(), gegner);
        }

        boolean[] valid = validMask(internesSpielfeld);
        double[] pi = policyFor(encode(internesSpielfeld, agentFarbe), valid);
        int a = greedyAction(pi, valid);

        int z = a / 3, s = a % 3;
        if (internesSpielfeld.getFarbe(z, s) != Farbe.Leer) {
            // Safe fallback: choose the first valid cell
            a = -1;
            for (int i = 0; i < 9; i++) {
                if (valid[i]) { a = i; break; }
            }
            if (a == -1) throw new IllegalerZugException(); // no valid moves (should be draw)
            z = a / 3; s = a % 3;
        }

        internesSpielfeld.setFarbe(z, s, agentFarbe);
        return new Zug(z, s);
    }

    @Override public void setName(String name) { this.name = name; }
    @Override public String getName() { return name; }
    @Override public void setFarbe(Farbe farbe) { this.agentFarbe = farbe; }
    @Override public Farbe getFarbe() { return agentFarbe; }

    /**
     * Speichert das trainierte Policy-Netzwerk.
     *
     * @param pfad Dateipfad zum Speichern
     * @throws IOException Bei Schreibfehlern
     */
    @Override
    public void speichereWissen(String pfad) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(pfad))) {
            oos.writeObject(policyNet);
        }
    }

    /**
     * Lädt ein zuvor gespeichertes Policy-Netzwerk.
     *
     * @param pfad Dateipfad zum Laden
     * @throws IOException Bei Lesefehlern
     */
    @Override
    public void ladeWissen(String pfad) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(pfad))) {
            this.policyNet = (NeuralNetwork<?>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Load error: " + e.getMessage());
        }
    }

    /**
     * Repräsentiert einen einzelnen Schritt in einer Episode.
     * Speichert alle Informationen für das Policy-Gradient Update.
     */
    private static class Step {
        /** Kodierter Spielzustand */
        final double[] state;

        /** Ausgeführte Aktion (0-8) */
        final int action;

        /** Erhaltener Reward (wird für Returns benötigt) */
        double reward;

        /** Maske gültiger Aktionen (für korrekte Normalisierung) */
        final boolean[] validMask;

        Step(double[] state, int action, double reward, boolean[] validMask) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.validMask = validMask;
        }
    }
}
