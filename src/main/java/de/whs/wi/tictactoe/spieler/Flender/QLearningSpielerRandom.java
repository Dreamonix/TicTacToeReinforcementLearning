package de.whs.wi.tictactoe.spieler.Flender;

import tictactoe.*;
import tictactoe.spieler.IAbbruchbedingung;
import tictactoe.spieler.ILernenderSpieler;
import tictactoe.spieler.beispiel.Zufallsspieler;

import java.io.*;
import java.util.Random;

public class QLearningSpielerRandom implements ILernenderSpieler {
   // --- Q-Matrix und Hyperparameter ---
    private static final int ANZAHL_ZUSTAENDE = 19683; // 3^9 mögliche Zustände
    private static final int ANZAHL_AKTIONEN = 9; // 3x3 Spielfeld

    // Q-Matrix: Zeilen=Zustände, Spalten=Aktionen
    private double[][] qMatrix = new double[ANZAHL_ZUSTAENDE][ANZAHL_AKTIONEN];

    // Lernparameter
    private static final double ALPHA = 0.2; // Lernrate
    private static final double GAMMA = 0.95; // Diskontfaktor
    private final double EPSILON_TRAINING_START = 1.0; // Startwert für Epsilon im Training
    private double epsilon = 1.0; // Explorationsrate
    private final double MIN_EPSILON = 0.01;

    private Random random = new Random();

    // ISpieler-Attribute
    private String agentName;
    private Farbe agentFarbe;
    private Spielfeld internesSpielfeld; // Kopie des Spielfelds für interne Zustandsverfolgung

    // --- Konstruktor ---
    public QLearningSpielerRandom(String name) {
        this.agentName = name;
        // Die Q-Matrix wird von Java standardmäßig mit 0.0 initialisiert
    }

    // --- 1. Zustands- und Aktionslogik ---

    /**
     * Konvertiert den 3x3-Zustand des Spielfelds in einen eindeutigen ternären Index (0 bis 19682).
     * @param internesSpielfeld Das aktuelle Spielfeld.
     * @return Der ternäre Index des Zustands.
     */
    public static int berechneZustandsIndex(Spielfeld internesSpielfeld) {
        int index = 0;
        int basis = 1; // 3^0

        // Iteriert von unten rechts (2,2) nach oben links (0,0)
        for (int zeile = 2; zeile >= 0; zeile--) {
            for (int spalte = 2; spalte >= 0; spalte--) {
                // toInt(): Leer=0, Kreis=1, Kreuz=2
                int farbWert = internesSpielfeld.getFarbe(zeile, spalte).toInt();
                index += farbWert * basis;
                basis *= 3; // Nächste Potenz von 3
            }
        }
        return index;
    }

    /**
     * Wählt die beste Aktion (Index 0-8) basierend auf der Epsilon-Greedy-Strategie.
     * @param internesSpielfeld Das aktuelle Spielfeld.
     * @return Der Index der gewählten Aktion.
     */
    private int waehleAktion(Spielfeld internesSpielfeld) {
        int sIndex = berechneZustandsIndex(internesSpielfeld);

        // Exploration (Zufälliger Zug)
        if (random.nextDouble() < epsilon) {
            int aktion;
            do {
                aktion = random.nextInt(ANZAHL_AKTIONEN); // 0 bis 8
            } while (internesSpielfeld.getFarbe(aktion / 3, aktion % 3) != Farbe.Leer);
            return aktion;
        }

        // Exploitation (Beste bekannte Aktion)
        else {
            int besteAktion = -1;
            double maxQ = -Double.MAX_VALUE;

            for (int aktion = 0; aktion < ANZAHL_AKTIONEN; aktion++) {
                int zeile = aktion / 3;
                int spalte = aktion % 3;

                // Nur leere Felder beachten
                if (internesSpielfeld.getFarbe(zeile, spalte) == Farbe.Leer) {
                    if (qMatrix[sIndex][aktion] > maxQ) {
                        maxQ = qMatrix[sIndex][aktion];
                        besteAktion = aktion;
                    }
                }
            }

            // Fallback, falls doch kein optimaler Zug gefunden wird (sollte bei Leerfeldern
            // in der Schleife oben aber immer einen Wert >= -1 liefern)
            if (besteAktion == -1) {
                // Wählt zufällig ein leeres Feld
                do {
                    besteAktion = random.nextInt(ANZAHL_AKTIONEN);
                } while (internesSpielfeld.getFarbe(besteAktion / 3, besteAktion % 3) != Farbe.Leer);
            }
            return besteAktion;
        }
    }

    /**
     * Führt den Q-Learning-Update-Schritt durch.
     */
    private void updateQ(Spielfeld s_alt, int aktion, double r, Spielfeld s_prime, Farbe agentFarbe) {
        int sIndex = berechneZustandsIndex(s_alt);
        int sPrimeIndex = berechneZustandsIndex(s_prime);

        // 1. Q(s, a)
        double q_old = qMatrix[sIndex][aktion];


        // 2. Prüfen, ob s' terminal ist (egal wer gewonnen hat / unentschieden
        Spielstand standAgent = s_prime.pruefeGewinn(agentFarbe);
        Spielstand standGegner = s_prime.pruefeGewinn(agentFarbe.opposite());

        // 3. max_a' Q(s', a') (Der höchste Q-Wert im Folgezustand)
        double maxQ_sPrime;


        if (standAgent != Spielstand.OFFEN || standGegner != Spielstand.OFFEN) {
            // Terminaler Zustand erreicht (Gewinn, Verlust oder Unentschieden): kein zukünftiger Wert
            maxQ_sPrime = 0.0;
        } else {
            // Nicht-terminal: max über alle LEGALEN Aktionen aus s'
            maxQ_sPrime = -Double.MAX_VALUE;
            for (int a_prime = 0; a_prime < ANZAHL_AKTIONEN; a_prime++) {
                int zeile = a_prime / 3;
                int spalte = a_prime % 3;
                // Nur legale Aktionen berücksichtigen
                if (s_prime.getFarbe(zeile, spalte) == Farbe.Leer) {
                    if (qMatrix[sPrimeIndex][a_prime] > maxQ_sPrime) {
                        maxQ_sPrime = qMatrix[sPrimeIndex][a_prime];
                    }
                }
            }
            // Falls keine legale Aktion gefunden (Bord voll), zukünftiger Wert = 0
            if (maxQ_sPrime == -Double.MAX_VALUE) {
                maxQ_sPrime = 0.0;
            }
        }

        // 3. Target-Berechnung: R + Gamma * max_a' Q(s', a')
        double qTarget = r + GAMMA * maxQ_sPrime;
        // 4. Q-Update: Q(s, a) <- Q(s, a) + Alpha * [Q_Target - Q(s, a)]
        qMatrix[sIndex][aktion] = q_old + ALPHA * (qTarget - q_old);
    }

    // --- 2. Trainingslogik (ILernenderSpieler) ---

    /**
     * @param abbruchBedingung Abbruchbedingung für das Training
     * @return true, wenn das Training erfolgreich abgeschlossen wurde
     */
    @Override
    public boolean trainieren(IAbbruchbedingung abbruchBedingung) {
        // 1. ZUERST Epsilon zurücksetzen, um neue Exploration zu erzwingen
        this.epsilon = EPSILON_TRAINING_START;
        long rundenZaehler = 0;

        // 2. Epsilon-Decay-Rate berechnen, die von 1.0 auf MIN_EPSILON über die ersten 500.000 Runden sinkt
        double epsilonDecayRunden = 1_000_000.0;
        double decayRate = Math.pow(MIN_EPSILON / epsilon, 1.0 / epsilonDecayRunden);

        while (!abbruchBedingung.abbruch()) {

            // Wechselnde Startspieler (z.B. 100.000 als Kreuz, 100.000 als Kreis)
            Farbe starterFarbe = (rundenZaehler % 2 == 0) ? Farbe.Kreuz : Farbe.Kreis;

            simuliereEinSpiel(starterFarbe);

            // Epsilon nur während der Decay-Phase anpassen
            if (rundenZaehler < epsilonDecayRunden && epsilon > MIN_EPSILON) {
                epsilon *= decayRate;
            } else {
                epsilon = MIN_EPSILON; // Sicherstellen, dass Epsilon nicht unter MIN_EPSILON fällt
            }

            if (rundenZaehler % 200_000 == 0) { // Ausgabe alle 20.000 Runden
                System.out.printf("Runde %d. Epsilon: %.4f%n",
                        rundenZaehler, epsilon);
            }
            rundenZaehler++;
        }
        return true;
    }

    /**
     * Simuliert ein einzelnes Spiel für das Training.
     * @param agentenStartFarbe Die Farbe des lernenden Spielers für dieses Spiel.
     */
    private void simuliereEinSpiel(Farbe agentenStartFarbe) {
        Spielfeld s = new Spielfeld();
        Zufallsspieler zufallsspieler = new Zufallsspieler("Gegner");

        Farbe agentenFarbe = agentenStartFarbe;
        Farbe gegnerFarbe = agentenFarbe.opposite();

        Farbe aktuelleFarbe = Farbe.Kreuz; // Kreuz beginnt immer
        Zug letzterGegnerZug = null;

        zufallsspieler.neuesSpiel(gegnerFarbe, 0);

        boolean spielLaeuft = true;

        while (spielLaeuft) {
            Spielfeld s_alt = s.clone();
            Zug neuerZug;

            if (aktuelleFarbe == agentenFarbe) {
                // Agentenzug
                int aktion = waehleAktion(s);
                int zeile = aktion / 3;
                int spalte = aktion % 3;
                neuerZug = new Zug(zeile, spalte);
                s.setFarbe(zeile, spalte, agentenFarbe);

                Spielfeld s_neu = s.clone();

                // Reward IMMER aus Sicht des Agenten
                double reward = 0.0;
                boolean spielEnde = false;

                Spielstand standAgent = s.pruefeGewinn(agentenFarbe);
                Spielstand standGegner = s.pruefeGewinn(gegnerFarbe);

                if (standAgent == Spielstand.GEWONNEN) {
                    reward = 1.0;
                    spielEnde = true;
                } else if (standGegner == Spielstand.GEWONNEN) {
                    reward = -1.0; // oder -1.5
                    spielEnde = true;
                } else if (spielfeldVoll(s)) {
                    reward = 0.0; // Remis neutral
                    spielEnde = true;
                }

                int aktionIndex = neuerZug.getZeile() * 3 + neuerZug.getSpalte();
                updateQ(s_alt, aktionIndex, reward, s_neu, agentenFarbe);

                if (spielEnde) {
                    break;
                }
            } else {
                // Gegnerzug (Random)
                zufallsspieler.setFarbe(gegnerFarbe);
                neuerZug = zufallsspieler.berechneZug(letzterGegnerZug, 0, 0);
                s.setFarbe(neuerZug.getZeile(), neuerZug.getSpalte(), gegnerFarbe);

                // Nach Gegnerzug nur auf Spielende prüfen, KEIN Q-Update
                Spielstand standAgent = s.pruefeGewinn(agentenFarbe);
                Spielstand standGegner = s.pruefeGewinn(gegnerFarbe);

                if (standAgent == Spielstand.GEWONNEN ||
                    standGegner == Spielstand.GEWONNEN ||
                    spielfeldVoll(s)) {
                    break;
                }
            }

            letzterGegnerZug = neuerZug;
            aktuelleFarbe = aktuelleFarbe.opposite();
        }
    }

    private boolean spielfeldVoll(Spielfeld s) {
        for (int z = 0; z < 3; z++) {
            for (int sp = 0; sp < 3; sp++) {
                if (s.getFarbe(z, sp) == Farbe.Leer) {
                    return false;
                }
            }
        }
        return true;
    }

    // --- 3. ISpieler Implementierung (für den Wettkampf) ---
    /**
     * Imitialisiert ein neues Spiel für den Lernenden Spieler.
     * @param agentFarbe
     * @param bedenkzeitInSekunden
     */
    @Override
    public void neuesSpiel(Farbe agentFarbe, int bedenkzeitInSekunden) {
        this.agentFarbe = agentFarbe;
        this.internesSpielfeld = new Spielfeld();
    }

    /**
     * @param vorherigerZug
     * @param l
     * @param l1
     * @return
     * @throws IllegalerZugException
     */
    @Override
    public Zug berechneZug(Zug vorherigerZug, long l, long l1) throws IllegalerZugException {
        // 1. Zug des Gegners auf unserem internen Spielfeld nachvollziehen
        if (vorherigerZug != null) {
            Farbe gegnerFarbe = agentFarbe.opposite();
            internesSpielfeld.setFarbe(vorherigerZug.getZeile(),
                    vorherigerZug.getSpalte(),
                    gegnerFarbe);
        }
        // 2. Beste Aktion im aktuellen (realen) Zustand wählen (epsilon=MIN_EPSILON, da Training beendet)
        double aktuellesEpsilon = this.epsilon;
        this.epsilon = 0.0; // Deaktiviere Exploration für den Wettkampf
        int aktion = waehleAktion(internesSpielfeld);
        this.epsilon = aktuellesEpsilon; // Epsilon für eventuelles Training wiederherstellen

        int zeile = aktion / 3;
        int spalte = aktion % 3;
        Zug neuerZug = new Zug(zeile, spalte);

        // 3. Zug auf unserem internen Spielfeld ausführen
        internesSpielfeld.setFarbe(neuerZug.getZeile(), neuerZug.getSpalte(), agentFarbe);

        return neuerZug;
    }

    // --- Weitere ISpieler Methoden ---
    @Override public void setName(String name) {this.agentName = name;}
    @Override public String getName() {return agentName;}
    @Override public void setFarbe(Farbe farbe) {this.agentFarbe = farbe;}
    @Override public Farbe getFarbe() {return agentFarbe;}

    // --- 4. Wissensspeicherung (ILernenderSpieler) ---
    /**
     * @param dateiPfad
     * @throws IOException
     */
    @Override
    public void speichereWissen(String dateiPfad) throws IOException {
        try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(dateiPfad))) {
            // 1. Array Speichern
            oos.writeObject(this.qMatrix);
            // 2. Epsilon speichern
            oos.writeDouble(this.epsilon); // Schreibt den primitiven double-Wert
            System.out.println("Q-Matrix gespeichert unter: " + dateiPfad);
        }
    }

    /**
     * @param dateiPfad
     * @throws IOException
     */
    @Override
    public void ladeWissen(String dateiPfad) throws IOException {
        try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(dateiPfad))) {
            // 1. Array laden muss als double[][] gelesen werden
            this.qMatrix = (double[][]) ois.readObject();
            // 2. Epsilon-Wert laden (Muss als double gelesen werden)
            this.epsilon = ois.readDouble();
            System.out.println("Q-Matrix erfolgreich geladen. Epsilon: " + this.epsilon);
        } catch (ClassNotFoundException e) {
            throw new IOException("Fehler beim Laden der Q-Matrix: " + e.getMessage());
        }
    }
}
