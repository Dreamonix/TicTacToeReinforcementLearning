package de.whs.wi.tictactoe.spieler.Flender;

import tictactoe.Farbe;
import tictactoe.spieler.AbbruchNachZeit;

import java.io.IOException;

public class TrainingRunner {
    public static void main(String[] args) {
        Spieler agent = new Spieler();
        agent.setName("Flender-QAgent");
        agent.setFarbe(Farbe.Kreis);

        // Optional: Load Previous Training
        try {
            agent.ladeWissen("wissen.bin");
            System.out.println("Loaded existing knowledge.");

        } catch (IOException e) {
            System.out.println("No existing knowledge found, starting fresh.");
        }

        AbbruchNachZeit abbruch = new AbbruchNachZeit(60); // 60 seconds training;
        System.out.println("Starting training...");
        try {
            boolean ok = agent.trainieren(abbruch);
            System.out.println("Training finished: " + ok);
        } catch (Exception e) {
            System.err.println("Training failed: " + e.getMessage());
            e.printStackTrace();
        }

        // Save Learned Knowledge
        try {
            agent.speichereWissen("wissen.bin");
            System.out.println("Knowledge saved.");
        } catch (IOException e) {
            System.err.println("Failed to save knowledge: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
