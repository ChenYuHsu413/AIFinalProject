/** Compact confusion matrix heatmap (cyan scale, dark-friendly). */
export function ConfusionMatrix({
  labels,
  matrix,
}: {
  labels: string[];
  matrix: number[][];
}) {
  const max = Math.max(1, ...matrix.flat());
  return (
    <div className="overflow-x-auto">
      <table className="text-xs">
        <thead>
          <tr>
            <th className="p-1 text-muted-foreground">真＼預</th>
            {labels.map((l) => (
              <th key={l} className="p-1 font-medium text-muted-foreground">
                {l}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td className="p-1 font-medium text-muted-foreground">{labels[i]}</td>
              {row.map((v, j) => {
                const a = v / max;
                return (
                  <td
                    key={j}
                    className="h-9 w-12 rounded text-center font-medium tabular-nums"
                    style={{
                      backgroundColor: `rgba(34,211,238,${a * 0.8})`,
                      color: a > 0.45 ? "#0b1220" : "#cbd5e1",
                    }}
                  >
                    {v}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-2 text-xs text-muted-foreground">
        列＝真實標籤，欄＝模型預測。對角線越集中越好。
      </p>
    </div>
  );
}
