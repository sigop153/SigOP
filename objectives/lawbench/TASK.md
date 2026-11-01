# LawBench charge prediction

Input: the facts section of a Chinese criminal indictment. Target: the charge or charges,
in Chinese. The executor answers inside `[罪名]...<eoa>`, several charges separated by `;`.
A prediction is correct only when its set of charges equals the target's; set F1 is recorded
alongside.

Facts are long and formal. Many charges have close neighbours (盗窃, 诈骗, 抢劫), and
multi-charge cases are common, so naming the main charge alone scores as wrong.
