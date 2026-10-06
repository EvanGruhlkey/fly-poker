import { createDemoTable, type TableView, type Card } from './model';
export interface DemoSession {
  readonly kind: 'playing' | 'finished';
  readonly street: number;
  readonly table: TableView;
  readonly history: readonly string[];
  readonly message: string;
}
export function createSession(handNumber = 1): DemoSession {
  return { kind: 'playing', street: 0, table: { ...createDemoTable(), handNumber },
    history: ['Blinds posted · 0.5 / 1 BB', 'You and the fly see the flop', 'Fly checks. Your move.'],
    message: 'Your move. What does a fly have up its sleeve?' };
}
export function act(state: DemoSession, action: 'fold' | 'call' | 'raise', amount: number): DemoSession {
  if (state.kind === 'finished') return state;
  if (action === 'fold') return { ...state, kind: 'finished',
    table: { ...state.table, flyStackBb: state.table.flyStackBb + state.table.potBb, potBb: 0 },
    history: [...state.history, 'You fold. Fly wins.'], message: 'You folded. The fly takes the pot.' };
  const paid = Math.min(state.table.playerStackBb, state.table.flyStackBb, action === 'call' ? 4 : Math.max(4, amount));
  const board: readonly Card[] = ['AS', '9D', '4C', 'TH', '2S'];
  const street = state.street + 1;
  const finished = street >= 3 || paid === state.table.playerStackBb;
  const pot = state.table.potBb + paid * 2;
  return { kind: finished ? 'finished' : 'playing', street,
    table: { ...state.table, board: board.slice(0, Math.min(5, street + 3)),
      playerStackBb: state.table.playerStackBb - paid, flyStackBb: state.table.flyStackBb - paid, potBb: pot,
      phase: finished ? 'showdown' : 'player-turn' },
    history: [...state.history, `You ${action === 'raise' ? 'bet' : 'call'} ${paid} BB. Fly calls.`,
      finished ? 'Demo complete. Deal another hand.' : `${street === 1 ? 'Turn' : 'River'} dealt. Fly checks.`],
    message: finished ? 'Demo hand complete. Deal again to keep playing.' : 'The fly calls, then checks. Your move.' };
}
