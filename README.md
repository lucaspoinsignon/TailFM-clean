```bash

if not torch.isfinite(loss):
            with torch.no_grad():
                pred = model(xt, t)
            print(f"\n  step {step}: NON-FINITE LOSS")
            for nm, v in (("x0", x0), ("x1", x1), ("xt", xt), ("pred", pred)):
                fin = v[torch.isfinite(v)]
                print(f"    {nm:4s} max|.| {fin.abs().max():.4g}  "
                      f"non-finite {int((~torch.isfinite(v)).sum())}")
            print(f"    t range [{t.min():.6f}, {t.max():.6f}]")
            bad = [n for n, p in model.named_parameters()
                   if not torch.isfinite(p).all()]
            print(f"    non-finite params: {bad[:5] or 'none'}")
            opt.zero_grad(set_to_none=True)
            continue


```
