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
 * Ein Q-Learning-basierter Tic-Tac-Toe-Spieler mit neuronalem Netzwerk (DQN-Style).
 *
 * <h2>Algorithmus-Übersicht</h2>
 * <p>Dieser Spieler implementiert Deep Q-Learning mit folgenden Kernkomponenten:</p>
 * <ul>
 *   <li><b>Experience Replay:</b> Speichert vergangene Erfahrungen und samplet zufällig für stabiles Training</li>
 *   <li><b>ε-greedy Exploration:</b> Balanciert Exploration und Exploitation mit abnehmender Explorationsrate</li>
 *   <li><b>Temporal Difference Learning:</b> Nutzt Bootstrap-Targets Q(s,a) = r + γ * max Q(s',a')</li>
 * </ul>
 *
 * <h2>Netzwerk-Architektur</h2>
 * <p>9 → 36 → 36 → 9 mit TANH-Aktivierung:</p>
 * <ul>
 *   <li><b>9 Inputs:</b> Spielfeld-Kodierung (-1 = Gegner, 0 = Leer, +1 = Agent)</li>
 *   <li><b>36-36 Hidden:</b> Ausreichend Kapazität für strategische Muster</li>
 *   <li><b>9 Outputs:</b> Q-Werte für jede mögliche Zug-Position</li>
 *   <li><b>TANH:</b> Ermöglicht negative Q-Werte (wichtig für Verlust-Signale)</li>
 * </ul>
 *
 * <h2>Hyperparameter</h2>
 * <ul>
 *   <li>GAMMA = 0.95: Discount-Faktor für zukünftige Rewards</li>
 *   <li>ALPHA = 0.01: Lernrate des neuronalen Netzwerks</li>
 *   <li>EPSILON: Startet bei 1.0, sinkt exponentiell auf 0.05</li>
 *   <li>REPLAY_BUFFER_SIZE = 10000: Maximale gespeicherte Erfahrungen</li>
 *   <li>BATCH_SIZE = 32: Anzahl Samples pro Trainingsschritt</li>
 * </ul>
 *
 * @author Flender
 * @see ILernenderSpieler
 */
public class QLearningNeurophSpieler implements ILernenderSpieler {

    /** Das neuronale Netzwerk zur Q-Wert-Approximation */
    private NeuralNetwork<?> qNetwork;

    /** Discount-Faktor: Gewichtung zukünftiger Rewards (0.95 = langfristig orientiert) */
    private static final double GAMMA = 0.95;

    /** Lernrate für Backpropagation (niedrig für Stabilität) */
    private static final double ALPHA = 0.01;

    /** Initiale Explorationsrate (100% zufällige Züge am Anfang) */
    private static final double EPSILON_START = 1.0;

    /** Minimale Explorationsrate (5% Zufall auch nach Training) */
    private static final double MIN_EPSILON = 0.05;

    /** Maximale Anzahl gespeicherter Erfahrungen im Replay Buffer */
    private static final int REPLAY_BUFFER_SIZE = 10000;

    /** Anzahl zufällig gesampelter Erfahrungen pro Trainingsschritt */
    private static final int BATCH_SIZE = 32;

    /** Mindestanzahl Erfahrungen bevor Training beginnt */
    private static final int MIN_REPLAY_SIZE = 500;

    /** FIFO-Buffer für Experience Replay */
    private final List<Experience> replayBuffer = new ArrayList<>();

    /** Zufallsgenerator für Exploration und Sampling */
    private final Random random = new Random();

    /** Aktuelle Explorationsrate (sinkt während des Trainings) */
    private double epsilon = EPSILON_START;

    /** Name des Spielers */
    private String name;

    /** Farbe des Agenten im aktuellen Spiel */
    private Farbe agentFarbe;

    /** Internes Spielfeld zur Zugberechnung */
    private Spielfeld internesSpielfeld;

    /**
     * Repräsentiert eine einzelne Erfahrung (State-Action-Reward-NextState Tupel).
     * Wird im Experience Replay Buffer gespeichert und für Batch-Training verwendet.
     */
    private static class Experience {
        /** Kodierter Spielzustand vor dem Zug */
        double[] state;

        /** Ausgeführte Aktion (0-8 = Feldposition) */
        int action;

        /** Erhaltene Belohnung (+1 Sieg, -1 Niederlage, 0.3 Unentschieden, 0 sonst) */
        double reward;

        /** Kodierter Spielzustand nach dem Zug */
        double[] nextState;

        /** True wenn Spiel beendet (keine weiteren Züge möglich) */
        boolean terminal;

        /** Maske der gültigen Aktionen im Folgezustand */
        boolean[] validNextActions;

        Experience(double[] state, int action, double reward, double[] nextState,
                   boolean terminal, boolean[] validNextActions) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.terminal = terminal;
            this.validNextActions = validNextActions;
        }
    }

    /**
     * Erstellt einen neuen Q-Learning Spieler mit initialisiertem Netzwerk.
     *
     * @param name Der Name des Spielers
     */
    public QLearningNeurophSpieler(String name) {
        this.name = name;
        initNetwork();
    }

    /**
     * Initialisiert das neuronale Netzwerk mit der Architektur 9-36-36-9.
     *
     * <p>Konfiguration:</p>
     * <ul>
     *   <li>TransferFunctionType.TANH für negative Q-Werte</li>
     *   <li>BackPropagation mit Lernrate ALPHA</li>
     *   <li>Single-Iteration pro learn() Aufruf</li>
     *   <li>Online-Modus (kein internes Batching)</li>
     * </ul>
     */
    private void initNetwork() {
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                TransferFunctionType.TANH,
                9, 36, 36, 9  // Larger hidden layers
        );

        org.neuroph.nnet.learning.BackPropagation bp =
                (org.neuroph.nnet.learning.BackPropagation) mlp.getLearningRule();
        bp.setLearningRate(ALPHA);
        bp.setMaxIterations(1);
        bp.setMaxError(Double.MAX_VALUE);
        bp.setBatchMode(false);
        this.qNetwork = mlp;
    }

    /**
     * Kodiert das Spielfeld als Eingabevektor für das neuronale Netzwerk.
     *
     * <p>Kodierungsschema:</p>
     * <ul>
     *   <li>+1.0: Eigene Farbe (Agent)</li>
     *   <li>-1.0: Gegnerische Farbe</li>
     *   <li>0.0: Leeres Feld</li>
     * </ul>
     *
     * @param board Das zu kodierende Spielfeld
     * @param agent Die Farbe des Agenten
     * @return double[9] Kodierter Zustandsvektor
     */
    private double[] encodeBoard(Spielfeld board, Farbe agent) {
        double[] input = new double[9];
        int idx = 0;
        for (int z = 0; z < 3; z++) {
            for (int s = 0; s < 3; s++) {
                Farbe f = board.getFarbe(z, s);
                if (f == Farbe.Leer) input[idx] = 0.0;
                else if (f == agent) input[idx] = 1.0;
                else input[idx] = -1.0;
                idx++;
            }
        }
        return input;
    }

    /**
     * Ermittelt die gültigen Aktionen (leere Felder) auf dem Spielfeld.
     *
     * @param board Das aktuelle Spielfeld
     * @return boolean[9] True für jedes leere Feld
     */
    private boolean[] getValidActions(Spielfeld board) {
        boolean[] valid = new boolean[9];
        for (int a = 0; a < 9; a++) {
            valid[a] = board.getFarbe(a / 3, a % 3) == Farbe.Leer;
        }
        return valid;
    }

    /**
     * Berechnet die Q-Werte für alle 9 Aktionen im gegebenen Zustand.
     *
     * @param stateInput Kodierter Spielzustand (9 Werte)
     * @return double[9] Q-Wert für jede Feldposition
     */
    private double[] predictQ(double[] stateInput) {
        qNetwork.setInput(stateInput);
        qNetwork.calculate();
        double[] out = new double[9];
        int i = 0;
        for (Double d : qNetwork.getOutput()) {
            out[i++] = d;
        }
        return out;
    }

    /**
     * Wählt eine Aktion basierend auf ε-greedy Policy.
     *
     * <p>Strategie:</p>
     * <ul>
     *   <li>Mit Wahrscheinlichkeit ε: Zufällige gültige Aktion</li>
     *   <li>Mit Wahrscheinlichkeit 1-ε: Aktion mit höchstem Q-Wert</li>
     * </ul>
     *
     * @param board Aktuelles Spielfeld
     * @param agent Farbe des Agenten
     * @return Gewählte Aktion (0-8)
     * @throws IllegalStateException Wenn keine gültigen Züge verfügbar
     */
    private int chooseAction(Spielfeld board, Farbe agent) {
        boolean[] valid = getValidActions(board);
        List<Integer> validActions = new ArrayList<>();
        for (int a = 0; a < 9; a++) {
            if (valid[a]) validActions.add(a);
        }

        // No valid actions - shouldn't happen in normal play
        if (validActions.isEmpty()) {
            throw new IllegalStateException("No valid actions available");
        }

        if (random.nextDouble() < epsilon) {
            return validActions.get(random.nextInt(validActions.size()));
        }

        double[] state = encodeBoard(board, agent);
        double[] qValues = predictQ(state);

        int bestAction = -1;
        double bestQ = -Double.MAX_VALUE;
        for (int a : validActions) {
            if (qValues[a] > bestQ) {
                bestQ = qValues[a];
                bestAction = a;
            }
        }

        // Fallback: if somehow bestAction is still -1, pick random valid
        if (bestAction == -1) {
            bestAction = validActions.get(random.nextInt(validActions.size()));
        }

        return bestAction;
    }

    /**
     * Speichert eine Erfahrung im Replay Buffer.
     *
     * <p>Bei vollem Buffer wird die älteste Erfahrung entfernt (FIFO).</p>
     *
     * @param sAlt Spielfeld vor dem Zug
     * @param action Ausgeführte Aktion
     * @param reward Erhaltene Belohnung
     * @param sNeu Spielfeld nach dem Zug
     * @param terminal True wenn Spiel beendet
     * @param agent Farbe des Agenten
     */
    private void storeExperience(Spielfeld sAlt, int action, double reward,
                                  Spielfeld sNeu, boolean terminal, Farbe agent) {
        double[] state = encodeBoard(sAlt, agent);
        double[] nextState = encodeBoard(sNeu, agent);
        boolean[] validNext = getValidActions(sNeu);

        Experience exp = new Experience(state, action, reward, nextState, terminal, validNext);

        if (replayBuffer.size() >= REPLAY_BUFFER_SIZE) {
            replayBuffer.remove(0);
        }
        replayBuffer.add(exp);
    }

    /**
     * Trainiert das Netzwerk mit einem zufälligen Batch aus dem Replay Buffer.
     *
     * <p>Algorithmus:</p>
     * <ol>
     *   <li>Sample BATCH_SIZE zufällige Erfahrungen</li>
     *   <li>Berechne Q-Targets: Q(s,a) = r + γ * max(Q(s',a')) für nicht-terminale Zustände</li>
     *   <li>Erstelle DataSet mit aktuellen Zuständen und modifizierten Q-Targets</li>
     *   <li>Trainiere Netzwerk mit Backpropagation</li>
     * </ol>
     *
     * <p>Training startet erst wenn MIN_REPLAY_SIZE Erfahrungen gesammelt wurden.</p>
     */
    private void trainFromReplay() {
        if (replayBuffer.size() < MIN_REPLAY_SIZE) return;

        // Sample random batch
        org.neuroph.core.data.DataSet batch = new org.neuroph.core.data.DataSet(9, 9);

        for (int i = 0; i < BATCH_SIZE; i++) {
            Experience exp = replayBuffer.get(random.nextInt(replayBuffer.size()));

            double[] qCurrent = predictQ(exp.state);
            double target;

            if (exp.terminal) {
                target = exp.reward;
            } else {
                double[] qNext = predictQ(exp.nextState);
                double maxNext = -Double.MAX_VALUE;
                for (int a = 0; a < 9; a++) {
                    if (exp.validNextActions[a] && qNext[a] > maxNext) {
                        maxNext = qNext[a];
                    }
                }
                if (maxNext == -Double.MAX_VALUE) maxNext = 0.0;
                target = exp.reward + GAMMA * maxNext;
            }

            double[] qTarget = qCurrent.clone();
            qTarget[exp.action] = target;
            batch.add(exp.state, qTarget);
        }

        qNetwork.learn(batch);
    }

    /**
     * Führt den Trainingsprozess durch bis die Abbruchbedingung erfüllt ist.
     *
     * <p>Training Loop:</p>
     * <ol>
     *   <li>Simuliere Spiel gegen Zufallsspieler (abwechselnd als Kreuz/Kreis)</li>
     *   <li>Speichere Erfahrungen im Replay Buffer</li>
     *   <li>Trainiere alle 4 Spiele vom Replay Buffer</li>
     *   <li>Reduziere ε exponentiell (Decay über 100.000 Runden)</li>
     * </ol>
     *
     * @param abbruch Abbruchbedingung (z.B. Zeitlimit oder Rundenanzahl)
     * @return true wenn Training erfolgreich
     */
    @Override
    public boolean trainieren(IAbbruchbedingung abbruch) {
        this.epsilon = EPSILON_START;
        long runde = 0;
        double decayRunden = 100_000.0;
        double decayRate = Math.pow(MIN_EPSILON / EPSILON_START, 1.0 / decayRunden);

        long startTime = System.currentTimeMillis();

        do {
            Farbe starter = (runde % 2 == 0) ? Farbe.Kreuz : Farbe.Kreis;
            simuliereSpielGegenZufall(starter);

            // Train every 4 games
            if (runde % 4 == 0) {
                trainFromReplay();
            }

            if (epsilon > MIN_EPSILON) {
                epsilon *= decayRate;
            }
            runde++;

            if (runde % 10000 == 0) {
                long elapsed = System.currentTimeMillis() - startTime;
                double gamesPerSec = runde / (elapsed / 1000.0);
                System.out.printf("Runde %d | eps=%.4f | buffer=%d | %.1f games/sec%n",
                        runde, epsilon, replayBuffer.size(), gamesPerSec);
            }

        } while (!abbruch.abbruch());

        return true;
    }

    /**
     * Simuliert ein komplettes Spiel gegen einen Zufallsspieler.
     *
     * <p>Sammelt Erfahrungen mit folgenden Rewards:</p>
     * <ul>
     *   <li>+1.0: Gewinn</li>
     *   <li>-1.0: Niederlage</li>
     *   <li>+0.3: Unentschieden (leicht positiv)</li>
     *   <li>0.0: Zwischenzüge</li>
     * </ul>
     *
     * @param agentStart Die Startfarbe des Agenten
     */
    private void simuliereSpielGegenZufall(Farbe agentStart) {
        Spielfeld board = new Spielfeld();
        tictactoe.spieler.beispiel.Zufallsspieler gegner =
                new tictactoe.spieler.beispiel.Zufallsspieler("Zufall-Gegner");

        Farbe agentColor = agentStart;
        Farbe oppColor = agentColor.opposite();
        Farbe amZug = Farbe.Kreuz;
        Zug letzterZug = null;

        gegner.neuesSpiel(oppColor, 0);

        Spielfeld lastAgentState = null;
        int lastAgentAction = -1;

        while (true) {
            if (amZug == agentColor) {
                Spielfeld sAlt = board.clone();
                int action = chooseAction(board, agentColor);
                board.setFarbe(action / 3, action % 3, agentColor);
                Spielfeld sNeu = board.clone();

                Spielstand stand = board.pruefeGewinn(agentColor);
                if (stand == Spielstand.GEWONNEN) {
                    storeExperience(sAlt, action, 1.0, sNeu, true, agentColor);
                    break;
                } else if (spielfeldVoll(board)) {
                    storeExperience(sAlt, action, 0.3, sNeu, true, agentColor);
                    break;
                }

                lastAgentState = sAlt;
                lastAgentAction = action;
                letzterZug = new Zug(action / 3, action % 3);

            } else {
                gegner.setFarbe(oppColor);
                Zug zug = gegner.berechneZug(letzterZug, 0, 0);
                board.setFarbe(zug.getZeile(), zug.getSpalte(), oppColor);
                letzterZug = zug;

                Spielstand standOpp = board.pruefeGewinn(oppColor);
                if (standOpp == Spielstand.GEWONNEN) {
                    if (lastAgentState != null) {
                        storeExperience(lastAgentState, lastAgentAction, -1.0,
                                board.clone(), true, agentColor);
                    }
                    break;
                } else if (spielfeldVoll(board)) {
                    if (lastAgentState != null) {
                        storeExperience(lastAgentState, lastAgentAction, 0.3,
                                board.clone(), true, agentColor);
                    }
                    break;
                } else if (lastAgentState != null) {
                    storeExperience(lastAgentState, lastAgentAction, 0.0,
                            board.clone(), false, agentColor);
                }
            }
            amZug = amZug.opposite();
        }
    }

    /**
     * Prüft ob das Spielfeld vollständig belegt ist (Unentschieden).
     *
     * @param s Das zu prüfende Spielfeld
     * @return true wenn alle 9 Felder belegt
     */
    private boolean spielfeldVoll(Spielfeld s) {
        for (int z = 0; z < 3; z++)
            for (int sp = 0; sp < 3; sp++)
                if (s.getFarbe(z, sp) == Farbe.Leer) return false;
        return true;
    }

    /**
     * Initialisiert den Spieler für ein neues Spiel.
     * Setzt ε auf 0 für rein greedy Spielweise im Wettkampf.
     *
     * @param farbe Die zugewiesene Farbe
     * @param bedenkzeitInSekunden Verfügbare Bedenkzeit (nicht verwendet)
     */
    @Override
    public void neuesSpiel(Farbe farbe, int bedenkzeitInSekunden) {
        this.agentFarbe = farbe;
        this.internesSpielfeld = new Spielfeld();
        this.epsilon = 0.0;
    }

    /**
     * Berechnet den nächsten Zug basierend auf der gelernten Q-Funktion.
     *
     * <p>Im Wettkampfmodus (ε=0) wird immer die Aktion mit dem
     * höchsten Q-Wert gewählt (greedy).</p>
     *
     * @param vorherigerZug Der letzte Zug des Gegners (null bei erstem Zug)
     * @param zeitKreis Verbleibende Zeit für Kreis
     * @param zeitKreuz Verbleibende Zeit für Kreuz
     * @return Der berechnete Zug
     * @throws IllegalerZugException Bei ungültigem Zug (sollte nicht auftreten)
     */
    @Override
    public Zug berechneZug(Zug vorherigerZug, long zeitKreis, long zeitKreuz) throws IllegalerZugException {
        if (vorherigerZug != null) {
            Farbe gegner = agentFarbe.opposite();
            internesSpielfeld.setFarbe(vorherigerZug.getZeile(), vorherigerZug.getSpalte(), gegner);
        }
        int action = chooseAction(internesSpielfeld, agentFarbe);
        int z = action / 3;
        int s = action % 3;
        if (internesSpielfeld.getFarbe(z, s) != Farbe.Leer) {
            throw new IllegalerZugException();
        }
        internesSpielfeld.setFarbe(z, s, agentFarbe);
        return new Zug(z, s);
    }

    @Override
    public void setName(String name) { this.name = name; }

    @Override
    public String getName() { return name; }

    @Override
    public void setFarbe(Farbe farbe) { this.agentFarbe = farbe; }

    @Override
    public Farbe getFarbe() { return agentFarbe; }

    /**
     * Setzt die Explorationsrate manuell (für Tests).
     * @param epsilon Neue Explorationsrate (0.0 - 1.0)
     */
    public void setEpsilon(double epsilon) { this.epsilon = epsilon; }

    /**
     * Speichert das trainierte Netzwerk mittels Java-Serialisierung.
     *
     * @param pfad Dateipfad zum Speichern
     * @throws IOException Bei Schreibfehlern
     */
    @Override
    public void speichereWissen(String pfad) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(pfad))) {
            oos.writeObject(qNetwork);
        }
    }

    /**
     * Lädt ein zuvor gespeichertes Netzwerk.
     *
     * @param pfad Dateipfad zum Laden
     * @throws IOException Bei Lesefehlern oder inkompatiblem Format
     */
    @Override
    public void ladeWissen(String pfad) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(pfad))) {
            this.qNetwork = (NeuralNetwork<?>) ois.readObject();
        } catch (ClassNotFoundException e) {
            throw new IOException("Fehler beim Laden: " + e.getMessage());
        }
    }
}
