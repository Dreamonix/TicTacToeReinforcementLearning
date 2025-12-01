package de.whs.wi.tictactoe.spieler.Flender;

import tictactoe.*;
import tictactoe.spieler.IAbbruchbedingung;
import tictactoe.spieler.ILernenderSpieler;

import java.io.*;
import java.util.*;

public class QSpieler implements ILernenderSpieler {
    private String spielerName;
    private Farbe spielerFarbe;
    private Spielfeld spielfeld;
    private Random random = new Random();

    // Q-Learning Parameters and Storage
    private final Map<String, double[]> qTable = new HashMap<>();
    private final double alpha = 0.1; // Learning rate
    private final double gamma = 0.9; // Discount factor
    private final double epsilon = 0.0; // Exploration rate

    private final List<StateAction> episodeHistory = new ArrayList<>();

    private static class StateAction implements Serializable {
        final String state;
        final int actionIndex;
        StateAction(String s, int a) {state = s; actionIndex = a;}
    }




    @Override
    public boolean trainieren(IAbbruchbedingung iAbbruchbedingung) {
        int episodes = 0;
        long startNs = System.nanoTime();
        if (this.spielerFarbe == null) this.spielerFarbe = Farbe.Kreis;

        while (!iAbbruchbedingung.abbruch()) {
            episodes++;
            // fresh training board and agent internal state
            Spielfeld s = new Spielfeld();
            this.episodeHistory.clear();
            this.neuesSpiel(this.spielerFarbe, 0);

            boolean agentStarts = random.nextBoolean(); // per-episode starter
            Zug lastOpponentMove = null;

            // play until terminal or no moves
            while (true) {
                // Agent's turn first this round
                if (agentStarts) {
                    // Agent plays (passed previous opponent move)
                    Zug agentMove;
                    try {
                        agentMove = this.berechneZug(lastOpponentMove, 0L, 0L);
                    } catch (IllegalerZugException e) {
                        break; // malformed agent move -> abort episode
                    }
                    s.setFarbe(agentMove.getZeile(), agentMove.getSpalte(), this.spielerFarbe);
                    // check terminal
                    if (s.pruefeGewinn(spielerFarbe) != Spielstand.OFFEN) break;

                    // Opponent random move
                    List<Integer> avail = availableActions(s);
                    if (avail.isEmpty()) break;
                    int opp = avail.get(random.nextInt(avail.size()));
                    s.setFarbe(opp / 3, opp % 3, this.spielerFarbe.opposite());
                    lastOpponentMove = new Zug(opp / 3, opp % 3);

                    // continue with agent starting next sub-turn
                    agentStarts = true;
                } else {
                    // Opponent starts this round
                    List<Integer> avail = availableActions(s);
                    if (avail.isEmpty()) break;
                    int opp = avail.get(random.nextInt(avail.size()));
                    s.setFarbe(opp / 3, opp % 3, this.spielerFarbe.opposite());
                    lastOpponentMove = new Zug(opp / 3, opp % 3);

                    // Agent responds
                    Zug agentMove;
                    try {
                        agentMove = this.berechneZug(lastOpponentMove, 0L, 0L);
                    } catch (IllegalerZugException e) {
                        break;
                    }
                    s.setFarbe(agentMove.getZeile(), agentMove.getSpalte(), this.spielerFarbe);
                    if (s.pruefeGewinn(spielerFarbe) != Spielstand.OFFEN) break;

                    // next round agent may or may not start; choose randomly per episode loop iteration
                    agentStarts = random.nextBoolean();
                }

                // If board full -> break
                if (availableActions(s).isEmpty()) break;
            }

            // compute reward from agent perspective
            double reward;
            if (s.pruefeGewinn(spielerFarbe) == Spielstand.GEWONNEN) reward = 1.0;
            else if (s.pruefeGewinn(spielerFarbe) == Spielstand.UNENTSCHIEDEN) reward = 0.0;
            else reward = -1.0;

            // update Q-table by going backwards over recorded episode
            List<StateAction> episode = new ArrayList<>(this.episodeHistory);
            double G = reward;
            for (int i = episode.size() - 1; i >= 0; i--) {
                StateAction sa = episode.get(i);
                double[] q = qTable.computeIfAbsent(sa.state, k -> new double[9]);
                double old = q[sa.actionIndex];
                q[sa.actionIndex] = old + alpha * (G - old);
                G = gamma * G;
            }
            this.episodeHistory.clear();
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
        if (vorherigerZug != null) {
            spielfeld.setFarbe(vorherigerZug.getZeile(),
                    vorherigerZug.getSpalte(),
                    spielerFarbe.opposite());
        }
        // choose action based on current board state
        String state = serializeState(spielfeld);
        int actionIndex = selectAction(state);
        int row = actionIndex / 3;
        int col = actionIndex % 3;

        // If chosen cell is not free, pick a random free cell (fallback)
        if (spielfeld.getFarbe(row, col) != Farbe.Leer) {
            List<Integer> avail = availableActions(spielfeld);
            if (avail.isEmpty()) throw new IllegalerZugException();
            {
                System.out.println("No available moves!");
            }
            int a = avail.get(random.nextInt(avail.size()));
            row = a / 3;
            col = a % 3;
            actionIndex = a;
        }

        // Perform move and record for training
        spielfeld.setFarbe(row, col, spielerFarbe);
        this.episodeHistory.add(new StateAction(state, actionIndex));
        return new Zug(row, col);
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

    public int getQTableSize() {
        synchronized (qTable) {
            return qTable.size();
        }
    }
}
