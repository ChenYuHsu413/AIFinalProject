"use client";

import { useEffect, useState } from "react";

import { apiGet, type Health, type ServoModelInfo } from "@/lib/api";

export function StatusBar() {
  const [health, setHealth] = useState<Health | null>(null);
  const [servo, setServo] = useState<ServoModelInfo | null>(null);
  const [err, setErr] = useState(false);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [h, s] = await Promise.all([
          apiGet<Health>("/health"),
          apiGet<ServoModelInfo>("/servo/model_info"),
        ]);
        if (!alive) return;
        setHealth(h);
        setServo(s);
      } catch {
        if (alive) setErr(true);
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const connected = !err && health !== null;

  return (
    <header className="flex h-12 items-center justify-between border-b bg-background px-5">
      <div className="flex items-center gap-2 text-sm">
        <span
          className={`h-2 w-2 rounded-full ${
            err ? "bg-red-500" : connected ? "bg-emerald-500" : "bg-amber-400"
          }`}
        />
        <span className="text-muted-foreground">
          {err ? "後端未連線" : connected ? "後端已連線" : "連線中…"}
        </span>
      </div>

      <div className="flex items-center gap-3 text-xs">
        {servo && (
          <>
            <span className="text-muted-foreground">
              主線模型 <span className="font-medium text-foreground">{servo.clf_model ?? "—"}</span>
              {servo.clf_macro_f1 != null && (
                <> · macro-F1 {servo.clf_macro_f1.toFixed(3)}</>
              )}
            </span>
            {servo.placeholder && (
              <span
                className="rounded bg-amber-100 px-2 py-0.5 font-medium text-amber-700"
                title="目前以 placeholder 合成資料訓練，非真實 PHM 伺服馬達資料"
              >
                placeholder 合成資料
              </span>
            )}
          </>
        )}
      </div>
    </header>
  );
}
