import { createDemoTable, type TableView, type Card } from './model';
export interface DemoSession {
 readonly kind: 'playing' | 'finished';
 readonly street: number;
 readonly toCall: number;
 readonly table: TableView;
 readonly history: readonly string[];
 readonly message: string;
}
export function createSession(handNumber = 1): DemoSession {
 return { kind: 'playing', street: 0, toCall: 4.25, table: { ...createDemoTable(), handNumber },
  history: ['Flop · A♠ 9♦ 4♣', 'Fly bets 4.25 BB. Your move.'],
  message: 'The fly bets 4.25 BB. Your move.' };
}
export function act(state: DemoSession, action: 'fold' | 'call' | 'raise', amount: number): DemoSession {
 if (state.kind === 'finished') return state;
 if (action === 'fold') return { ...state, kind: 'finished',
  table: { ...state.table, flyStackBb: state.table.flyStackBb + state.table.potBb, potBb: 0 },
  history: [...state.history, 'You fold. Fly wins.'], message: 'You folded. The fly takes the pot.' };
 const paid = Math.min(state.table.playerStackBb, state.table.flyStackBb, action === 'call' ? state.toCall : Math.max(state.toCall + 1, amount));
 const flyPaid = Math.max(0, paid - state.toCall);
 const board: readonly Card[] = ['AS', '9D', '4C', 'TH', '2S'];
 const street = state.street + 1;
 const finished = street >= 3 || paid === state.table.playerStackBb || flyPaid === state.table.flyStackBb;
 const pot = state.table.potBb + paid + flyPaid;
 return { kind: finished ? 'finished' : 'playing', street, toCall: 0,
  table: { ...state.table, board: board.slice(0, finished ? 5 : Math.min(5, street + 3)),
   playerStackBb: state.table.playerStackBb - paid, flyStackBb: state.table.flyStackBb - flyPaid + (finished ? pot : 0), potBb: finished ? 0 : pot,
   phase: finished ? 'showdown' : 'player-turn' },
  history: [...state.history, paid === 0 ? 'You check. Fly checks.' : `You ${action === 'raise' ? 'raise to' : 'call'} ${paid} BB.${flyPaid > 0 ? ' Fly calls.' : ''}`,
   finished ? 'Fly shows A♥ 8♠. Pair of aces wins.' : `${street === 1 ? 'Turn' : 'River'} dealt. Fly checks.`],
  message: finished ? 'The fly wins with a pair of aces. Scripted demo hand.' : 'The fly checks. Check back or make a bet.' };
}
