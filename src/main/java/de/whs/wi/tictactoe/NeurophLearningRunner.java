package de.whs.wi.tictactoe;

import de.whs.wi.tictactoe.spieler.Flender.HeuristikSpieler;
import de.whs.wi.tictactoe.spieler.Flender.QLearningNeurophSpieler;
import de.whs.wi.tictactoe.spieler.Flender.QLearningSpielerHeuristik;
import tictactoe.TicTacToe;
import tictactoe.spieler.AbbruchNachIterationen;
import tictactoe.spieler.ILernenderSpieler;
import tictactoe.spieler.ISpieler;
import tictactoe.spieler.beispiel.Zufallsspieler;

import java.io.IOException;

public class NeurophLearningRunner {
    public static void main(String[] args) {
        ISpieler zufaelligerSpieler = new Zufallsspieler("Zufall");
        ILernenderSpieler agent = new QLearningNeurophSpieler("Flender-Neuroph-Agent");

        // --- Wissen zuerst laden ---
        try {
            agent.ladeWissen("wissenNeurophZufall.bin");
            System.out.println("Vorhandenes Wissen gegen Zufallsspieler-Spieler geladen.");
        } catch (IOException e) {
            System.out.println("Kein vorhandenes Wissen gefunden, starte neu.");
        }
        System.out.println("===================================================================");

        TicTacToe spiel = new TicTacToe();
        ISpieler gewinner;
        int gewinneZufall;
        int gewinneAgent;
        int unentschieden;
        double trainingIterations = 1e6;

        // Evaluate Before Training (Dies ist nun die Evaluierung des geladenen Zustands)
        gewinneZufall = 0;
        gewinneAgent = 0;
        System.out.println("Status vor dem Training:");
        System.out.println(zufaelligerSpieler.getName() + " vs. " + agent.getName());
        System.out.println("=========================================================");
        for (int i = 0; i < 1000; i++) {
            gewinner = spiel.neuesSpiel(zufaelligerSpieler, agent, 150, false);
            if (gewinner == zufaelligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;

            gewinner = spiel.neuesSpiel(agent, zufaelligerSpieler, 150, false);
            if (gewinner == zufaelligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;
        }
        System.out.println("Gewinne " + zufaelligerSpieler.getName() + ": " + gewinneZufall);
        System.out.println("Gewinne " + agent.getName() + ": " + gewinneAgent);
        System.out.println("=========================================================");

        // Training Phase (iterations)
        System.out.printf("Starte Training mit %d Iterationen. Bitte warten...", ((int)trainingIterations));
        long startTime = System.currentTimeMillis();
        agent.trainieren(new AbbruchNachIterationen((int) trainingIterations));
        if (agent instanceof QLearningNeurophSpieler qAgent) {
            qAgent.setEpsilon(0.0); // Setze Epsilon nach dem Training auf einen niedrigen Wert
        }
        long endTime = System.currentTimeMillis();
        System.out.println("Training beendet. Gesamtdauer in Sekunden: " + ((endTime - startTime) / 1000.0));

        // Evaluate After Training
        gewinneZufall = 0;
        gewinneAgent = 0;
        unentschieden = 0;
        System.out.println("Nach dem Training:");
        System.out.println(zufaelligerSpieler.getName() + " vs. " + agent.getName());
        System.out.println("=========================================================");
        for (int i = 0; i < 1000; i++) {
            gewinner = spiel.neuesSpiel(zufaelligerSpieler, agent, 150, false);
            if (gewinner == zufaelligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;
            else unentschieden++;

            gewinner = spiel.neuesSpiel(agent, zufaelligerSpieler, 150, false);
            if (gewinner == zufaelligerSpieler) gewinneZufall++;
            else if (gewinner == agent) gewinneAgent++;
            else unentschieden++;
        }
        System.out.println("Gewinne " + zufaelligerSpieler.getName() + ": " + gewinneZufall);
        System.out.println("Unentschieden: " + unentschieden);
        System.out.println("Gewinne " + agent.getName() + ": " + gewinneAgent);
        System.out.println("=========================================================");

        // Save Learned Knowledge
        try {
            agent.speichereWissen("wissenNeurophZufall.bin");
            System.out.println("Wissen gegen Zufalls-Spieler gespeichert.");
        } catch (IOException e) {
            System.err.println("Fehler beim Speichern des Wissens: " + e.getMessage());
        }
    }
}
