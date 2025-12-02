// java
// File: `src/main/java/de/whs/wi/tictactoe/spieler/Flender/TrainingRunner.java`
package de.whs.wi.tictactoe;

import de.whs.wi.tictactoe.spieler.Flender.QLearningSpielerRandom;
import tictactoe.TicTacToe;
import tictactoe.spieler.AbbruchNachIterationen;
import tictactoe.spieler.ILernenderSpieler;
import tictactoe.spieler.ISpieler;
import tictactoe.spieler.beispiel.Zufallsspieler;

import java.io.IOException;

public class QLearningTrainingRunner {
    public static void main(String[] args) {
        ISpieler zufalligerSpieler = new Zufallsspieler("Zufall");
        ILernenderSpieler agent = new QLearningSpielerRandom("Flender-QLearner-Agent");

        // --- Wissen zuerst laden ---
        try {
            agent.ladeWissen("wissenZufall.bin");
            System.out.println("Vorhandenes Wissen gegen Zufallsspieler geladen.");
        } catch (IOException e) {
            System.out.println("Kein vorhandenes Wissen gefunden, starte neu.");
        }
        System.out.println("===================================================================");

        TicTacToe spiel = new TicTacToe();
        ISpieler gewinner;
        int gewinneZufall;
        int gewinneAgent;
        int unentschieden;
        double trainingIterations = 5e7;

        // Evaluate Before Training (Dies ist nun die Evaluierung des geladenen Zustands)
        gewinneZufall = 0;
        gewinneAgent = 0;
        System.out.println("Status vor dem Training:");
        System.out.println(zufalligerSpieler.getName() + " vs. " + agent.getName());
        System.out.println("=========================================================");
        for (int i = 0; i < 1000; i++) {
            gewinner = spiel.neuesSpiel(zufalligerSpieler, agent, 150, false);
            if (gewinner == zufalligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;

            gewinner = spiel.neuesSpiel(agent, zufalligerSpieler, 150, false);
            if (gewinner == zufalligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;
        }
        System.out.println("Gewinne " + zufalligerSpieler.getName() + ": " + gewinneZufall);
        System.out.println("Gewinne " + agent.getName() + ": " + gewinneAgent);
        System.out.println("=========================================================");

        // Training Phase (iterations)
        System.out.printf("Starte Training mit %d Iterationen. Bitte warten...", ((int)trainingIterations));
        long startTime = System.currentTimeMillis();
        agent.trainieren(new AbbruchNachIterationen((int) trainingIterations));
        long endTime = System.currentTimeMillis();
        System.out.println("Training beendet. Gesamtdauer in Sekunden: " + ((endTime - startTime) / 1000.0));

        // Evaluate After Training
        gewinneZufall = 0;
        gewinneAgent = 0;
        unentschieden = 0;
        System.out.println("Nach dem Training:");
        System.out.println(zufalligerSpieler.getName() + " vs. " + agent.getName());
        System.out.println("=========================================================");
        for (int i = 0; i < 1000; i++) {
            gewinner = spiel.neuesSpiel(zufalligerSpieler, agent, 150, false);
            if (gewinner == zufalligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;
            else unentschieden++;

            gewinner = spiel.neuesSpiel(agent, zufalligerSpieler, 150, false);
            if (gewinner == zufalligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;
            else unentschieden++;
        }
        System.out.println("Gewinne " + zufalligerSpieler.getName() + ": " + gewinneZufall);
        System.out.println("Unentschieden: " + unentschieden);
        System.out.println("Gewinne " + agent.getName() + ": " + gewinneAgent);
        System.out.println("=========================================================");

        // Save Learned Knowledge
        try {
            agent.speichereWissen("wissenZufall.bin");
            System.out.println("Wissen gegem Zufallsspieler gespeichert.");
        } catch (IOException e) {
            System.err.println("Fehler beim Speichern des Wissens: " + e.getMessage());
        }
    }
}