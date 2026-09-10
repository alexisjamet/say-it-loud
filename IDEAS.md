# Idées de fonctionnalités (backlog)

Notées le 2026-09-10, à reprendre plus tard. Par ordre d'impact estimé.

1. **Insertion directe au curseur (Mac).** Coller automatiquement le texte dans
   l'app active (CGEvent ⌘V, permission Accessibilité) au lieu du presse-papiers.
2. **Push-to-talk.** Maintenir la touche pour parler, relâcher pour insérer.
3. **Nettoyage sans LLM.** Supprimer les « euh », « hum » ; dictionnaire perso de
   remplacements (noms propres, jargon).
4. **Transcrire des fichiers audio / Dictaphone.** Glisser un m4a sur la fenêtre,
   extension de partage iOS.
5. **Sous-titres du son système (Mac).** ScreenCaptureKit pour les visios, puis
   « résume cette réunion » avec le LLM.
6. **Intégrations Apple.** App Intents (Raccourcis, bouton Action), widget écran
   verrouillé, recherche dans l'historique, export Markdown.
7. **Synchro iCloud de l'historique** entre iPhone et Mac (CloudKit).

## Post-traitement LLM (en cours)

- 2026-09-10 : Qwen3 1.7B jugé insuffisant, Qwen3 4B « pas très bon » en
  français. Ministral 3 3B (`mlx-community/Ministral-3-3B-Instruct-2512-4bit`)
  ajouté et mis par défaut ; jugé « bien mieux », Qwen retiré complètement.
- Le prompt inclut un exemple (one-shot) par preset, dans la langue du texte :
  sans lui Ministral ajoute des notes, du markdown et invente des détails.
- Pistes suivantes si ce n'est toujours pas assez bon : Ministral 3 8B (Mac
  seulement, environ 4,5 Go), Gemma 4 E4B (portage lourd, voir discussion),
  ou Claude via l'API avec une clé fournie par l'utilisateur (opt-in, le texte
  quitte l'appareil ; environ 0,005 $ par reformulation avec Opus 5).
- Presets : corriger, court pour un pote, soigné pour un email, liste à puces,
  traduire en anglais / français, consigne libre.
- iPhone : décharger le modèle STT avant le LLM (RAM).
