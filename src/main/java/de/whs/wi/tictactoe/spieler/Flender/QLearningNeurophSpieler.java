package de.whs.wi.tictactoe.spieler.Flender;

import org.neuroph.core.NeuralNetwork;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.util.TransferFunctionType;
import tictactoe.*;
import tictactoe.spieler.IAbbruchbedingung;
import tictactoe.spieler.ILernenderSpieler;

import java.io.*;
import java.util.Random;

public class QLearningNeurophSpieler implements ILernenderSpieler {


    private NeuralNetwork<?> qNetwork;

    private static final double GAMMA = 0.99; // Discount factor
    private static final double ALPHA = 0.01; // Learning rate
    private static final double EPSILON_START = 1.0; // Initial exploration rate
    private static final double MIN_EPSILON = 0.1; // Minimum exploration rate

    private double epsilon = EPSILON_START;
    private final Random random = new Random();

    private String name;
    private Farbe agentFarbe;
    private Spielfeld internesSpielfeld;

    public QLearningNeurophSpieler(String name) {
        this.name = name;
        initNetwork();
    }

    private void initNetwork() {
        // 9-18-18-9 MLP Network mit Tanh Aktivierungsfunktion
        MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                TransferFunctionType.TANH,
                9, 18, 18, 9
        );

        mlp.getLearningRule().setLearningRate(ALPHA);
        this.qNetwork = mlp;
    }


    // --- Encoding: Spielfeld -> double[9] ---
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

    // Vorwärtslauf: Zustand -> Q-Werte für 9 Aktionen
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

    // Epsilon-greedy Action-Selection auf Basis der Netz-Q-Werte
    private int chooseAction(Spielfeld board, Farbe agent) {
        // Exploration
        if (random.nextDouble() < epsilon) {
            int action;
            do {
                action = random.nextInt(9);
            } while (board.getFarbe(action / 3, action % 3) != Farbe.Leer);
            return action;
        }

        // Exploitation
        double[] state = encodeBoard(board, agent);
        double[] qValues = predictQ(state);

        int bestAction = -1;
        double bestQ = -Double.MAX_VALUE;
        for (int a = 0; a < 9; a++) {
            int z = a / 3;
            int s = a % 3;
            if (board.getFarbe(z, s) != Farbe.Leer) continue;
            if (qValues[a] > bestQ) {
                bestQ = qValues[a];
                bestAction = a;
            }
        }

        // Fallback zufällige legale Aktion
        if (bestAction == -1) {
            int action;
            do {
                action = random.nextInt(9);
            } while (board.getFarbe(action / 3, action % 3) != Farbe.Leer);
            return action;
        }
        return bestAction;
    }

    // Ein Q-Learning-Update mit dem Netz durchführen
    private void trainOnTransition(Spielfeld sAlt,
                                   int action,
                                   double reward,
                                   Spielfeld sNeu,
                                   boolean terminal,
                                   Farbe agent) {

        double[] state = encodeBoard(sAlt, agent);
        double[] nextState = encodeBoard(sNeu, agent);

        double[] qCurrent = predictQ(state);
        double target;

        if (terminal) {
            target = reward;
        } else {
            double[] qNext = predictQ(nextState);
            double maxNext = -Double.MAX_VALUE;
            for (int a = 0; a < 9; a++) {
                int z = a / 3;
                int sp = a % 3;
                if (sNeu.getFarbe(z, sp) != Farbe.Leer) continue;
                if (qNext[a] > maxNext) maxNext = qNext[a];
            }
            if (maxNext == -Double.MAX_VALUE) maxNext = 0.0;
            target = reward + GAMMA * maxNext;
        }

        // Zielvektor erstellen
        double[] qTarget = qCurrent.clone();
        qTarget[action] = target;

        // Neuroph: Online-Training mit DataSet (1 Sample)
        org.neuroph.core.data.DataSet trainingSet =
                new org.neuroph.core.data.DataSet(9, 9);
        trainingSet.addRow(new org.neuroph.core.data.DataSetRow(state, qTarget));

        // Ein Trainingsschritt
        qNetwork.learn(trainingSet);
    }

    // --- Trainingslogik: wie bei deinem tabellarischen Agenten ---

    @Override
    public boolean trainieren(IAbbruchbedingung abbruch) {
        this.epsilon = EPSILON_START;
        long runde = 0;
        double decayRunden = 1_000_000.0;
        double decayRate = Math.pow(MIN_EPSILON / EPSILON_START, 1.0 / decayRunden);

        while (!abbruch.abbruch()) {
            Farbe starter = (runde % 2 == 0) ? Farbe.Kreuz : Farbe.Kreis;
            simuliereSpielGegenHeuristik(starter);

            if (runde < decayRunden && epsilon > MIN_EPSILON) {
                epsilon *= decayRate;
            } else {
                epsilon = MIN_EPSILON;
            }

            if (runde % 200_000 == 0) {
                System.out.printf("NN-Q Runde %d, epsilon=%.4f%n", runde, epsilon);
            }
            runde++;
        }
        return true;
    }

    private void simuliereSpielGegenHeuristik(Farbe agentStart) {
        Spielfeld board = new Spielfeld();
        HeuristikSpieler gegner = new HeuristikSpieler("Heuristik-Gegner");

        Farbe agentColor = agentStart;
        Farbe oppColor = agentColor.opposite();

        Farbe amZug = Farbe.Kreuz; // Kreuz beginnt
        Zug letzterZug = null;

        gegner.neuesSpiel(oppColor, 0);

        while (true) {
            Spielfeld sAlt = board.clone();
            Zug zug;

            if (amZug == agentColor) {
                int action = chooseAction(board, agentColor);
                int z = action / 3;
                int s = action % 3;
                zug = new Zug(z, s);
                board.setFarbe(z, s, agentColor);

                Spielfeld sNeu = board.clone();

                Spielstand standAgent = board.pruefeGewinn(agentColor);
                Spielstand standGegner = board.pruefeGewinn(oppColor);

                double reward = 0.0;
                boolean terminal = false;

                if (standAgent == Spielstand.GEWONNEN) {
                    reward = 1.0;
                    terminal = true;
                } else if (standGegner == Spielstand.GEWONNEN) {
                    reward = -2.0;
                    terminal = true;
                } else if (spielfeldVoll(board)) {
                    reward = 0.5;
                    terminal = true;
                }

                trainOnTransition(sAlt, action, reward, sNeu, terminal, agentColor);

                letzterZug = zug;
                if (terminal) break;
            } else {
                try {
                    zug = gegner.berechneZug(letzterZug, 0, 0);
                } catch (IllegalerZugException e) {
                    int a;
                    int z, s;
                    do {
                        a = random.nextInt(9);
                        z = a / 3;
                        s = a % 3;
                    } while (board.getFarbe(z, s) != Farbe.Leer);
                    zug = new Zug(z, s);
                }
                board.setFarbe(zug.getZeile(), zug.getSpalte(), oppColor);
                letzterZug = zug;

                Spielstand standAgent = board.pruefeGewinn(agentColor);
                Spielstand standGegner = board.pruefeGewinn(oppColor);

                if (standAgent == Spielstand.GEWONNEN
                        || standGegner == Spielstand.GEWONNEN
                        || spielfeldVoll(board)) {
                    break;
                }
            }

            amZug = amZug.opposite();
        }
    }

    private boolean spielfeldVoll(Spielfeld s) {
        for (int z = 0; z < 3; z++)
            for (int sp = 0; sp < 3; sp++)
                if (s.getFarbe(z, sp) == Farbe.Leer) return false;
        return true;
    }

    // --- ISpieler-Teil für Wettkampf ---

    @Override
    public void neuesSpiel(Farbe farbe, int bedenkzeitInSekunden) {
        this.agentFarbe = farbe;
        this.internesSpielfeld = new Spielfeld();
        // Im Wettkampf: keine Exploration
        this.epsilon = 0.0;
    }

    @Override
    public Zug berechneZug(Zug vorherigerZug, long zeitKreis, long zeitKreuz) throws IllegalerZugException {
        if (vorherigerZug != null) {
            Farbe gegner = agentFarbe.opposite();
            internesSpielfeld.setFarbe(
                    vorherigerZug.getZeile(),
                    vorherigerZug.getSpalte(),
                    gegner
            );
        }
        int action = chooseAction(internesSpielfeld, agentFarbe);
        int z = action / 3;
        int s = action % 3;
        if (internesSpielfeld.getFarbe(z, s) != Farbe.Leer) {
            throw new IllegalerZugException(); // Ohne Parameter
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

    // --- Wissensspeicherung für Netz ---

    @Override
    public void speichereWissen(String pfad) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(pfad))) {
            oos.writeObject(qNetwork);
            System.out.println("NN-Q Wissen gespeichert unter: " + pfad);
        }
    }

    @Override
    public void ladeWissen(String pfad) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(pfad))) {
            this.qNetwork = (NeuralNetwork<?>) ois.readObject();
            System.out.println("NN-Q Wissen geladen aus: " + pfad);
        } catch (ClassNotFoundException e) {
            throw new IOException("Fehler beim Laden des NN: " + e.getMessage());
        }
    }
}

