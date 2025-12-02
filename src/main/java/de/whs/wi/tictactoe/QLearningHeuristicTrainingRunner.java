// java
// File: `src/main/java/de/whs/wi/tictactoe/spieler/Flender/TrainingRunner.java`
package de.whs.wi.tictactoe;

import de.whs.wi.tictactoe.spieler.Flender.QLearningSpielerHeuristik;
import de.whs.wi.tictactoe.spieler.Flender.HeuristikSpieler;
import tictactoe.TicTacToe;
import tictactoe.spieler.AbbruchNachIterationen;
import tictactoe.spieler.ILernenderSpieler;
import tictactoe.spieler.ISpieler;

import java.io.IOException;

public class QLearningHeuristicTrainingRunner {
    public static void main(String[] args) {
        ISpieler heuristikSpieler = new HeuristikSpieler("Heuristik");
        ILernenderSpieler agent = new QLearningSpielerHeuristik("Flender-QLearner-Agent");

        // --- Wissen zuerst laden ---
        try {
            agent.ladeWissen("wissenHeuristik.bin");
            System.out.println("Vorhandenes Wissen gegen Heuristik-Spieler geladen.");
        } catch (IOException e) {
            System.out.println("Kein vorhandenes Wissen gefunden, starte neu.");
        }
        System.out.println("===================================================================");

        TicTacToe spiel = new TicTacToe();
        ISpieler gewinner;
        int gewinneHeuristik;
        int gewinneAgent;
        int unentschieden;
        double trainingIterations = 1e8;

        // Evaluate Before Training (Dies ist nun die Evaluierung des geladenen Zustands)
        gewinneHeuristik = 0;
        gewinneAgent = 0;
        System.out.println("Status vor dem Training:");
        System.out.println(heuristikSpieler.getName() + " vs. " + agent.getName());
        System.out.println("=========================================================");
        for (int i = 0; i < 1000; i++) {
            gewinner = spiel.neuesSpiel(heuristikSpieler, agent, 150, false);
            if (gewinner == heuristikSpieler) gewinneHeuristik++;
            else if (gewinner == agent) gewinneAgent++;

            gewinner = spiel.neuesSpiel(agent, heuristikSpieler, 150, false);
            if (gewinner == heuristikSpieler) gewinneHeuristik++;
            else if (gewinner == agent) gewinneAgent++;
        }
        System.out.println("Gewinne " + heuristikSpieler.getName() + ": " + gewinneHeuristik);
        System.out.println("Gewinne " + agent.getName() + ": " + gewinneAgent);
        System.out.println("=========================================================");

        // Training Phase (iterations)
        System.out.printf("Starte Training mit %d Iterationen. Bitte warten...", ((int)trainingIterations));
        long startTime = System.currentTimeMillis();
        agent.trainieren(new AbbruchNachIterationen((int) trainingIterations));
        if (agent instanceof QLearningSpielerHeuristik qAgent) {
            qAgent.setEpsilon(0.0); // Setze Epsilon nach dem Training auf einen niedrigen Wert
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Training beendet. Gesamtdauer in Sekunden: " + ((endTime - startTime) / 1000.0));

        // Evaluate After Training
        gewinneHeuristik = 0;
        gewinneAgent = 0;
        unentschieden = 0;
        System.out.println("Nach dem Training:");
        System.out.println(heuristikSpieler.getName() + " vs. " + agent.getName());
        System.out.println("=========================================================");
        for (int i = 0; i < 1000; i++) {
            gewinner = spiel.neuesSpiel(heuristikSpieler, agent, 150, false);
            if (gewinner == heuristikSpieler) gewinneHeuristik++;
            else if (gewinner == agent) gewinneAgent++;
            else unentschieden++;

            gewinner = spiel.neuesSpiel(agent, heuristikSpieler, 150, false);
            if (gewinner == heuristikSpieler) gewinneHeuristik++;
            else if (gewinner == agent) gewinneAgent++;
            else unentschieden++;
        }
        System.out.println("Gewinne " + heuristikSpieler.getName() + ": " + gewinneHeuristik);
        System.out.println("Unentschieden: " + unentschieden);
        System.out.println("Gewinne " + agent.getName() + ": " + gewinneAgent);
        System.out.println("=========================================================");

        // Save Learned Knowledge
        try {
            agent.speichereWissen("wissenHeuristik.bin");
            System.out.println("Wissen gegen Heuristik-Spieler gespeichert.");
        } catch (IOException e) {
            System.err.println("Fehler beim Speichern des Wissens: " + e.getMessage());
        }
    }
}