import { describe, expect, it } from "vitest";

import { createDemoTable } from "../game/model";
import { hudMarkup } from "./hud";

describe("hudMarkup", () => {
  it("shows the hand, legal actions, and connectome activity", () => {
    const markup = hudMarkup(createDemoTable());

    expect(markup).toContain("Poker vs. a Fruit Fly");
    expect(markup).toContain("Call 4.25 BB");
    expect(markup).toContain("12,482");
    expect(markup).toContain("Live connectome");
  });
});
