package de.whs.wi.tictactoe.spieler.Flender;

import tictactoe.*;
import tictactoe.spieler.ISpieler;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class HeuristikSpieler implements ISpieler {

    private String name;
    private Farbe farbe;
    private Spielfeld spielfeld;
    private Random random = new Random();

    public HeuristikSpieler(String name) {
        this.name = name;
    }

    @Override
    public void neuesSpiel(Farbe farbe, int bedenkzeitInSekunden) {
        this.farbe = farbe;
        this.spielfeld = new Spielfeld();
    }

    @Override
    public Zug berechneZug(Zug vorherigerZug, long zeitKreis, long zeitKreuz) throws IllegalerZugException {
        // 1\. Gegenzug ins interne Spielfeld eintragen
        if (vorherigerZug != null) {
            spielfeld.setFarbe(
                    vorherigerZug.getZeile(),
                    vorherigerZug.getSpalte(),
                    farbe.opposite()
            );
        }

        // 2\. Alle leeren Felder sammeln
        List<Zug> leereFelder = new ArrayList<>();
        for (int z = 0; z < 3; z++) {
            for (int s = 0; s < 3; s++) {
                if (spielfeld.getFarbe(z, s) == Farbe.Leer) {
                    leereFelder.add(new Zug(z, s));
                }
            }
        }

        // 3\. Gewinnzug suchen
        for (Zug zug : leereFelder) {
            Spielfeld kopie = spielfeld.clone();
            kopie.setFarbe(zug.getZeile(), zug.getSpalte(), farbe);
            if (kopie.pruefeGewinn(farbe) == Spielstand.GEWONNEN) {
                spielfeld.setFarbe(zug.getZeile(), zug.getSpalte(), farbe);
                return zug;
            }
        }

        // 4\. Blockzug gegen direkten Gewinn des Gegners
        Farbe gegner = farbe.opposite();
        for (Zug zug : leereFelder) {
            Spielfeld kopie = spielfeld.clone();
            kopie.setFarbe(zug.getZeile(), zug.getSpalte(), gegner);
            if (kopie.pruefeGewinn(gegner) == Spielstand.GEWONNEN) {
                spielfeld.setFarbe(zug.getZeile(), zug.getSpalte(), farbe);
                return zug;
            }
        }

        // 5\. Sonst zufälliges leeres Feld
        Zug zug = leereFelder.get(random.nextInt(leereFelder.size()));
        spielfeld.setFarbe(zug.getZeile(), zug.getSpalte(), farbe);
        return zug;
    }

    @Override public void setName(String name) { this.name = name; }
    @Override public String getName() { return name; }
    @Override public void setFarbe(Farbe farbe) { this.farbe = farbe; }
    @Override public Farbe getFarbe() { return farbe; }
}