package de.whs.wi.tictactoe.spieler.Flender;

import tictactoe.*;
import tictactoe.spieler.IAbbruchbedingung;
import tictactoe.spieler.ILernenderSpieler;

import java.io.*;
import java.util.*;

public class Spieler implements ILernenderSpieler {
    private String spielerName;
    private Farbe spielerFarbe;
    private Spielfeld spielfeld;
    private Random random = new Random();

    // Q-Learning Parameters and Storage
    private final Map<String, double[]> qTable = new HashMap<>();
    private final double alpha = 0.1; // Learning rate
    private final double gamma = 0.9; // Discount factor
    private final double epsilon = 0.2; // Exploration rate

    private final List<StateAction> episodeHistory = new ArrayList<>();

    private static class StateAction implements Serializable {
        final String state;
        final int actionIndex;
        StateAction(String s, int a) {state = s; actionIndex = a;}
    }




    @Override
    public boolean trainieren(IAbbruchbedingung iAbbruchbedingung) {
        // Simple self-play training loop until the abort condition signals stop.
        // The IAbbruchbedingung interface is provided by the JAR; here we poll it.
        // For each episode we simulate a game between two agents (this agent vs random),
        // record the episode and perform a simple Monte-Carlo style update on Q-values.
        int episodes = 0;
        while (!iAbbruchbedingung.abbruch()) {
            episodes++;
            Spielfeld s = new Spielfeld();
            Farbe currentPlayer = Farbe.Kreis;
            List<StateAction> episode = new ArrayList<>();
            // play until terminal
            while (s.pruefeGewinn(spielerFarbe) == Spielstand.OFFEN) {
                String state = serializeState(s);
                int action = selectActionForState(state, s);
                int row = action / 3;
                int col = action % 3;
                s.setFarbe(row, col, currentPlayer);
                episode.add(new StateAction(state, action));
                if (s.pruefeGewinn(spielerFarbe) != Spielstand.OFFEN) break;
                // opponent random move
                List<Integer> avail = availableActions(s);
                if (avail.isEmpty()) break;
                int opp = avail.get(random.nextInt(avail.size()));
                s.setFarbe(opp / 3, opp % 3, currentPlayer.opposite());
            }
            // reward from perspective of this player
            double reward;
            if (s.pruefeGewinn(spielerFarbe) == Spielstand.GEWONNEN) reward = 1.0;
            else if (s.pruefeGewinn(spielerFarbe) == Spielstand.UNENTSCHIEDEN) reward = 0.0;
            else reward = -1.0;
            // simple backward update (Monte-Carlo)
            double G = reward;
            for (int i = episode.size() - 1; i >= 0; i--) {
                StateAction sa = episode.get(i);
                double[] q = qTable.computeIfAbsent(sa.state, k -> new double[9]);
                double old = q[sa.actionIndex];
                q[sa.actionIndex] = old + alpha * (G - old);
                G = gamma * G;
            }
            // allow aborting after many episodes if needed
        }
        return true;
    }

    @Override
    public void speichereWissen(String s) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(s))) {
            oos.writeObject(qTable);
        }
    }

    @Override
    public void ladeWissen(String s) throws IOException {
        File f = new File(s);
        if (!f.exists()) return;
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(f))) {
            Object obj = ois.readObject();
            if (obj instanceof Map<?, ?> loaded) {
                // unchecked cast but expected structure
                qTable.clear();
                for (Map.Entry<?,?> e : loaded.entrySet()) {
                    if (e.getKey() instanceof String && e.getValue() instanceof double[]) {
                        qTable.put((String) e.getKey(), (double[]) e.getValue());
                    }
                }
            }
        } catch (ClassNotFoundException ex) {
            throw new IOException(ex);
        }

    }

    @Override
    public void neuesSpiel(Farbe farbe, int bedenkzeitInSekunden) {
        this.spielerFarbe = farbe;
        this.spielfeld = new Spielfeld();
    }

    @Override
    public String getName() {
        return this.spielerName;
    }

    @Override
    public void setName(String s) {
        this.spielerName = s;
    }

    @Override
    public Farbe getFarbe() {
        return this.spielerFarbe;
    }

    @Override
    public void setFarbe(Farbe farbe) {
        this.spielerFarbe = farbe;
    }

    @Override
    public Zug berechneZug(Zug vorherigerZug, long x, long y) throws IllegalerZugException {
        //Zunächst den vorherigen Zug des Gegners durchführen
        if (vorherigerZug != null)
            spielfeld.setFarbe(vorherigerZug.getZeile(),
                    vorherigerZug.getSpalte(),
                    spielerFarbe.opposite());
            Zug neuerZug;
            do {
                neuerZug = new Zug(random.nextInt(3), random.nextInt(3));
            }
            while (spielfeld.getFarbe(neuerZug.getZeile(), neuerZug.getSpalte()) != Farbe.Leer);
            spielfeld.setFarbe(neuerZug.getZeile(), neuerZug.getSpalte(), spielerFarbe);
            return neuerZug;

    }


    // Helper methods for Q-Learning
    private String serializeState(Spielfeld s) {
        StringBuilder sb = new StringBuilder(9);
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                Farbe f = s.getFarbe(r, c);
                sb.append((char) ('0' + f.ordinal()));
            }
        }
        return sb.toString();
    }

    private List<Integer> availableActions(Spielfeld s) {
        List<Integer> list = new ArrayList<>();
        for (int i = 0; i < 9; i++) {
            int r = i / 3, c = i % 3;
            if (s.getFarbe(r, c) == Farbe.Leer) list.add(i);
        }
        return list;
    }

    private int selectActionForState(String state, Spielfeld s) {
        // helper used for training self-play: acts on a provided Spielfeld
        double[] q = qTable.computeIfAbsent(state, k -> new double[9]);
        List<Integer> avail = availableActions(s);
        if (avail.isEmpty()) return 0;
        if (random.nextDouble() < epsilon) {
            return avail.get(random.nextInt(avail.size()));
        } else {
            int best = avail.get(0);
            double bestVal = q[best];
            for (int a : avail) {
                if (q[a] > bestVal) { best = a; bestVal = q[a]; }
            }
            return best;
        }
    }

    private int selectAction(String state) {
        double[] q = qTable.computeIfAbsent(state, k -> new double[9]);
        List<Integer> avail = new ArrayList<>();
        for (int i = 0; i < 9; i++) {
            int r = i / 3, c = i % 3;
            if (spielfeld.getFarbe(r, c) == Farbe.Leer) avail.add(i);
        }
        if (avail.isEmpty()) return 0;
        if (random.nextDouble() < epsilon) {
            return avail.get(random.nextInt(avail.size()));
        } else {
            int best = avail.get(0);
            double bestVal = q[best];
            for (int a : avail) {
                if (q[a] > bestVal) { best = a; bestVal = q[a]; }
            }
            return best;
        }
    }
}
