document.addEventListener('DOMContentLoaded', async () => {
    const calendarEl = document.getElementById('calendar');
    if (!calendarEl) return;

    // 1. 서비스 목록 불러오기
    try {
        const serviceResponse = await fetch('/api/services');
        const services = await serviceResponse.json();
        const checkboxContainer = document.getElementById('serviceCheckboxes');
        if (checkboxContainer) {
            services.forEach(s => {
                const div = document.createElement('div');
                div.className = 'form-check form-check-inline mb-2';
                div.innerHTML = `
                    <input class="form-check-input service-checkbox" type="checkbox" id="svc_${s.id}" value="${s.id}" data-duration="${s.duration}">
                    <label class="form-check-label" for="svc_${s.id}">
                        <span class="badge" style="background-color: ${s.color || '#0d6efd'}; color: white;">${s.name} (${s.duration}분)</span>
                    </label>
                `;
                checkboxContainer.appendChild(div);
            });
        }
    } catch (error) { console.error('서비스 목록 로드 실패:', error); }

    // 2. FullCalendar 초기화
    const calendar = new FullCalendar.Calendar(calendarEl, {
        initialView: 'timeGridWeek',
        locale: 'ko',
        headerToolbar: { left: 'prev,next today', center: 'title', right: 'dayGridMonth,timeGridWeek' },
        selectable: true,
        slotMinTime: '00:00:00',
        slotMaxTime: '24:00:00',
        events: '/api/reservations',
        
        // 날짜/시간 선택 시 가용 시간 슬롯 조회
        select: async function(info) {
            const dateStr = info.startStr.split('T')[0]; // YYYY-MM-DD
            
            // 과거 날짜 선택 방지
            if (new Date(dateStr) < new Date().setHours(0,0,0,0)) {
                alert('과거 날짜는 예약할 수 없음.');
                calendar.unselect();
                return;
            }

            // 서버에서 가용 시간 조회
            try {
                const res = await fetch(`/api/available-times?date=${dateStr}`);
                const data = await res.json();
                
                const container = document.getElementById('timeSlotsContainer');
                if (!container) {
                    console.error("예약 모달 내 #timeSlotsContainer 요소를 찾을 수 없음.");
                    return;
                }

                // 3. 시간 버튼 UI 렌더링 (체크 아이콘 제거됨)
                container.innerHTML = data.slots
                    .filter(s => {
                        const hour = parseInt(s.time.split(':')[0], 10);
                        return hour < 3 || hour >= 5; // 03~05시 제외
                    })
                    .map(s => `
                        <button type="button" 
                                class="time-slot-btn btn ${s.available ? 'btn-outline-primary' : 'btn-secondary disabled'} m-1 d-inline-flex align-items-center justify-content-center" 
                                style="transition: all 0.2s ease; min-width: 95px; border-width: 2px;"
                                data-time="${dateStr} ${s.time}"
                                ${s.available ? `onclick="selectTime(this, '${dateStr} ${s.time}')"` : 'disabled'}
                                aria-pressed="false">
                            <span class="time-text">${s.time}</span>
                        </button>
                    `).join('');
                
                new bootstrap.Modal(document.getElementById('reservationModal')).show();
            } catch (e) {
                console.error("시간 정보 로드 실패:", e);
                alert('시간 정보를 불러오는 중 오류 발생.');
            }
        }
    });
    calendar.render();

    // 4. 예약 제출 로직
    const saveBtn = document.getElementById('saveReservation');
    if (saveBtn) {
        saveBtn.addEventListener('click', async () => {
            const data = {
                customer_name: document.getElementById('customerName').value,
                customer_phone: document.getElementById('customerPhone').value,
                car_number: document.getElementById('carNumber').value,
                start_time: document.getElementById('startTime').value, 
                service_ids: Array.from(document.querySelectorAll('.service-checkbox:checked')).map(cb => parseInt(cb.value))
            };

            const response = await fetch('/api/reservations', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const result = await response.json();
            if (result.success) {
                alert('예약이 확정되었습니다.');
                bootstrap.Modal.getInstance(document.getElementById('reservationModal')).hide();
                calendar.refetchEvents();
            } else {
                alert('예약 실패: ' + (result.message || '알 수 없는 오류'));
            }
        });
    }
});

// 5. 시간 슬롯 클릭 시 시각적 피드백 토글 및 데이터 저장 (아이콘 관련 코드 제거됨)
window.selectTime = (btnElement, fullDateTime) => {
    // 5-1. 기존 선택된 버튼들의 활성화 상태 초기화
    document.querySelectorAll('.time-slot-btn').forEach(btn => {
        if (!btn.classList.contains('disabled')) {
            btn.classList.remove('btn-primary', 'text-white', 'shadow-sm');
            btn.classList.add('btn-outline-primary');
            btn.setAttribute('aria-pressed', 'false');
        }
    });

    // 5-2. 클릭된 요소에 브랜드 컬러(primary) 배경 및 글씨 색상 적용
    btnElement.classList.remove('btn-outline-primary');
    btnElement.classList.add('btn-primary', 'text-white', 'shadow-sm');
    btnElement.setAttribute('aria-pressed', 'true');

    // 5-3. 폼 전송을 위한 hidden input에 날짜/시간 저장
    document.getElementById('startTime').value = fullDateTime;
};