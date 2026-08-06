# Multilingual natural-language examples

The local parser does not send prompts to an external service.

```bash
# Chinese
python vibe_agent.py "比较 Cu、Ag 和 Pt(111) 上 C、CH、CH2、CH3 的吸附，固定底部两层" --execute

# English
python vibe_agent.py "Benchmark C, CH, CH2 and CH3 on Cu, Ag and Pd(111) with constrained relaxation"

# Spanish
python vibe_agent.py "Calcular C, CH2 y CH3 sobre cobre, plata y paladio (111)"

# French
python vibe_agent.py "Comparer C, CH et CH3 sur cuivre, argent et platine (111)"

# German
python vibe_agent.py "Berechne C und CH3 auf Kupfer, Silber und Platin (111)"

# Japanese / formula-led input
python vibe_agent.py "Cu、Ag、Au(111)で C、CH2、CH3 を比較"

# Automatic elemental-metal modelling: fcc, bcc, hcp
python vibe_agent.py "Calculate CO adsorption on Ni(100)" --execute
python vibe_agent.py "计算CO在Fe(110)上的吸附能" --execute
python vibe_agent.py "计算CHO在Co(0001)上的吸附能" --execute
python vibe_agent.py "计算CO在Ru(10-10)上的吸附能" --execute
```

Without `--execute`, the command only prints and saves a validated JSON plan.
Add `--execute` to calculate and `--yes` to skip confirmation.
