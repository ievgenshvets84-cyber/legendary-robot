/* LLM-Forge – minimaler Canvas-Liniendiagramm-Zeichner.
   Keine externe Bibliothek: zeichnet Trainings- und Validierungsverlust. */

class LineChart {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext("2d");
        this.series = {}; // { name: {points: [{x,y}], color} }
        this._resize();
        window.addEventListener("resize", () => { this._resize(); this.draw(); });
    }

    _resize() {
        // Für scharfe Darstellung an Gerätepixel anpassen
        const ratio = window.devicePixelRatio || 1;
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight || 220;
        this.canvas.width = width * ratio;
        this.canvas.height = height * ratio;
        this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
        this.w = width;
        this.h = height;
    }

    addSeries(name, color) {
        this.series[name] = { points: [], color };
    }

    push(name, x, y) {
        if (y === null || y === undefined || Number.isNaN(y)) return;
        if (!this.series[name]) this.addSeries(name, "#4c9aff");
        this.series[name].points.push({ x, y });
    }

    reset() {
        for (const key of Object.keys(this.series)) this.series[key].points = [];
        this.draw();
    }

    draw() {
        const ctx = this.ctx;
        ctx.clearRect(0, 0, this.w, this.h);
        const pad = { l: 46, r: 12, t: 12, b: 26 };

        // Wertebereich über alle Serien bestimmen
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        let hasData = false;
        for (const s of Object.values(this.series)) {
            for (const p of s.points) {
                hasData = true;
                minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
                minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
            }
        }
        if (!hasData) {
            ctx.fillStyle = "#9aa7b5";
            ctx.font = "13px sans-serif";
            ctx.fillText("Noch keine Trainingsdaten …", pad.l, this.h / 2);
            return;
        }
        if (minX === maxX) maxX = minX + 1;
        if (minY === maxY) { minY -= 0.5; maxY += 0.5; }

        const plotW = this.w - pad.l - pad.r;
        const plotH = this.h - pad.t - pad.b;
        const sx = (x) => pad.l + ((x - minX) / (maxX - minX)) * plotW;
        const sy = (y) => pad.t + (1 - (y - minY) / (maxY - minY)) * plotH;

        // Gitter und Y-Achsenbeschriftung
        ctx.strokeStyle = "#2d3648";
        ctx.fillStyle = "#9aa7b5";
        ctx.font = "11px sans-serif";
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = pad.t + (plotH * i) / 4;
            const val = maxY - ((maxY - minY) * i) / 4;
            ctx.beginPath();
            ctx.moveTo(pad.l, y); ctx.lineTo(this.w - pad.r, y); ctx.stroke();
            ctx.fillText(val.toFixed(2), 6, y + 4);
        }

        // Serien zeichnen
        for (const s of Object.values(this.series)) {
            if (s.points.length === 0) continue;
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            s.points.forEach((p, i) => {
                const X = sx(p.x), Y = sy(p.y);
                if (i === 0) ctx.moveTo(X, Y); else ctx.lineTo(X, Y);
            });
            ctx.stroke();
        }

        // Legende
        let lx = pad.l;
        ctx.font = "12px sans-serif";
        for (const [name, s] of Object.entries(this.series)) {
            if (s.points.length === 0) continue;
            ctx.fillStyle = s.color;
            ctx.fillRect(lx, this.h - 14, 12, 12);
            ctx.fillStyle = "#e6edf3";
            ctx.fillText(name, lx + 16, this.h - 4);
            lx += ctx.measureText(name).width + 40;
        }
    }
}
